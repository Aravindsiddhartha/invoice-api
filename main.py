import re
import os
import json
import requests
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser as dateparser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    invoice_text: str


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You extract structured data from raw invoice text. Return ONLY a JSON object with exactly these 6 keys, no other text, no markdown fences:

{
  "invoice_no": string or null,
  "date": string in YYYY-MM-DD format or null,
  "vendor": string or null,
  "amount": number or null,
  "tax": number or null,
  "currency": one of "USD","EUR","GBP","INR","JPY" or null
}

Rules:
- amount is the SUBTOTAL / pre-tax amount, NOT the grand total. If only a total and a tax amount are given, compute amount = total - tax.
- tax is the tax amount only (as a number), not a percentage.
- date must be normalized to ISO format YYYY-MM-DD regardless of the input format.
- currency should be inferred from symbols (₹, $, €, £, ¥) or words (rupees, dollars, euros, pounds, yen) or explicit ISO codes.
- Use null for any field that cannot be determined from the text.
- Return ONLY the JSON object, nothing else."""


def extract_via_llm(text: str):
    if not GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)

        result = {
            "invoice_no": data.get("invoice_no"),
            "date": data.get("date"),
            "vendor": data.get("vendor"),
            "amount": data.get("amount"),
            "tax": data.get("tax"),
            "currency": data.get("currency"),
        }
        # basic sanity checks
        if result["amount"] is not None:
            result["amount"] = float(result["amount"])
        if result["tax"] is not None:
            result["tax"] = float(result["tax"])
        if result["date"]:
            try:
                datetime.strptime(result["date"], "%Y-%m-%d")
            except ValueError:
                result["date"] = None
        return result
    except Exception:
        return None


CURRENCY_SYMBOLS = {
    "₹": "INR", "rs.": "INR", "rs ": "INR", "inr": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pounds": "GBP", "sterling": "GBP",
    "¥": "JPY", "jpy": "JPY", "yen": "JPY",
}

NUM_RE = r"[\d,]+(?:\.\d+)?"


def clean_number(s: str):
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def find_label_value(text: str, labels, stop_at_newline=True, strict=False):
    """Find 'Label: value' style lines for any of the given label patterns,
    anchored to the start of a line (ignoring leading whitespace).

    strict=True adds a guard so the label can't match if immediately followed
    by another word (e.g. 'Tax ID:' must NOT match a bare 'tax' pattern, and
    'Total Pages:' must NOT match a bare 'total' pattern) -- only a separator,
    digit, or currency symbol may follow. Use strict=True for generic/bare
    fallback labels prone to colliding with compound phrases; leave it False
    for specific labels (e.g. invoice number formats without a colon).
    """
    for label in labels:
        if strict:
            pattern = rf"^[ \t]*{label}[ \t]*(?=[:\-]|[ \t]*[\d$€£¥₹])[:\-]?[ \t]*(.+)$"
        else:
            pattern = rf"^[ \t]*{label}[ \t]*[:\-]?[ \t]*(.+)$"
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if stop_at_newline:
                val = val.split("\n")[0].strip()
            if val:
                return val
    return None


def extract_invoice_no(text: str):
    val = find_label_value(text, [
        r"invoice\s*(?:no\.?|number|#)",
        r"inv\s*(?:no\.?|#)",
    ])
    if val:
        m = re.match(r"([A-Za-z0-9\-\/]+)", val)
        if m:
            return m.group(1).strip().rstrip(".,;")
    # fallback: look for standalone INV-style code anywhere
    m = re.search(r"\b([A-Z]{2,}-\d{2,}-?\d*)\b", text)
    if m:
        return m.group(1)
    return None


def extract_date(text: str):
    val = find_label_value(text, [
        r"(?:invoice\s*)?date",
        r"dated",
    ])
    candidates = []
    if val:
        candidates.append(val)
    # also scan whole text for date-like substrings as fallback
    date_regexes = [
        r"\d{4}-\d{2}-\d{2}",       # ISO first, unambiguous
        r"\d{1,2}\s+\w+\s+\d{4}",
        r"\w+\s+\d{1,2},?\s+\d{4}",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
    ]
    for dr in date_regexes:
        for m in re.finditer(dr, text):
            candidates.append(m.group(0))

    for cand in candidates:
        iso_match = re.match(r"^\d{4}-\d{2}-\d{2}$", cand.strip())
        try:
            if iso_match:
                dt = dateparser.parse(cand, fuzzy=True)  # unambiguous, no dayfirst needed
            else:
                dt = dateparser.parse(cand, fuzzy=True, dayfirst=True)
            if dt:
                return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            continue
    return None


def extract_vendor(text: str):
    val = find_label_value(text, [
        r"vendor",
        r"bill(?:ed)?\s*from",
        r"seller",
        r"from",
        r"company",
        r"supplier",
    ])
    if val:
        # strip trailing punctuation and stray labels
        val = re.sub(r"[.,;]+$", "", val).strip()
        return val
    return None


def extract_amount_and_tax(text: str):
    subtotal = None
    tax = None
    total = None

    subtotal_val = find_label_value(text, [
        r"sub\s*-?\s*total",
        r"taxable\s*(?:value|amount)",
        r"base\s*amount",
        r"net\s*amount",
        r"amount\s*\(excl\.?\s*(?:tax|gst|vat)\)",
        r"amount\s*before\s*tax",
        r"amount\s*excl\.?\s*(?:tax|gst|vat)",
        r"invoice\s*amount",
        r"invoice\s*value",
        r"order\s*value",
        r"bill(?:ed)?\s*amount",
        r"item\s*total",
        r"goods\s*value",
        r"principal\s*amount",
        r"amount",
        r"price",
        r"value",
        r"cost",
    ], strict=True)
    if subtotal_val:
        m = re.search(NUM_RE, subtotal_val)
        if m:
            subtotal = clean_number(m.group(0))

    tax_val = find_label_value(text, [
        r"gst\s*\([\d.]+%\)",
        r"gst",
        r"vat\s*\([\d.]+%\)",
        r"vat",
        r"sales\s*tax",
        r"tax\s*\([\d.]+%\)",
        r"tax\s*amount",
        r"total\s*tax",
        r"tax",
    ], strict=True)
    if tax_val:
        m = re.search(NUM_RE, tax_val)
        if m:
            tax = clean_number(m.group(0))

    total_val = find_label_value(text, [
        r"grand\s*total",
        r"total\s*amount\s*due",
        r"amount\s*due",
        r"total\s*payable",
        r"total\s*amount",
        r"total",
    ], strict=True)
    if total_val:
        m = re.search(NUM_RE, total_val)
        if m:
            total = clean_number(m.group(0))

    # If subtotal missing but we have total and tax, derive it
    if subtotal is None and total is not None and tax is not None:
        subtotal = round(total - tax, 2)

    return subtotal, tax


def extract_currency(text: str):
    # symbol-based, most reliable
    if "₹" in text:
        return "INR"
    if "$" in text:
        return "USD"
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    if "¥" in text:
        return "JPY"

    # word-boundary based checks to avoid matching "rs" inside "Traders" etc.
    checks = [
        (r"\bRs\.?\b", "INR"), (r"\bINR\b", "INR"), (r"\brupees\b", "INR"),
        (r"\bUSD\b", "USD"), (r"\bdollars\b", "USD"),
        (r"\bEUR\b", "EUR"), (r"\beuros\b", "EUR"),
        (r"\bGBP\b", "GBP"), (r"\bpounds\b", "GBP"), (r"\bsterling\b", "GBP"),
        (r"\bJPY\b", "JPY"), (r"\byen\b", "JPY"),
    ]
    for pattern, code in checks:
        if re.search(pattern, text, re.IGNORECASE):
            return code
    return None


@app.post("/extract")
def extract(req: InvoiceRequest):
    text = req.invoice_text

    llm_result = extract_via_llm(text)
    if llm_result is not None:
        # fill any nulls from regex as a safety net
        regex_invoice_no = extract_invoice_no(text)
        regex_date = extract_date(text)
        regex_vendor = extract_vendor(text)
        regex_amount, regex_tax = extract_amount_and_tax(text)
        regex_currency = extract_currency(text)

        return {
            "invoice_no": llm_result["invoice_no"] or regex_invoice_no,
            "date": llm_result["date"] or regex_date,
            "vendor": llm_result["vendor"] or regex_vendor,
            "amount": llm_result["amount"] if llm_result["amount"] is not None else regex_amount,
            "tax": llm_result["tax"] if llm_result["tax"] is not None else regex_tax,
            "currency": llm_result["currency"] or regex_currency,
        }

    # LLM unavailable or failed -- pure regex fallback
    invoice_no = extract_invoice_no(text)
    date = extract_date(text)
    vendor = extract_vendor(text)
    amount, tax = extract_amount_and_tax(text)
    currency = extract_currency(text)

    return {
        "invoice_no": invoice_no,
        "date": date,
        "vendor": vendor,
        "amount": amount,
        "tax": tax,
        "currency": currency,
    }


@app.get("/")
def health():
    return {"status": "ok"}
