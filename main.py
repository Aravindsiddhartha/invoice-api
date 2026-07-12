import re
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


def find_label_value(text: str, labels, stop_at_newline=True):
    """Find 'Label: value' style lines for any of the given label patterns."""
    for label in labels:
        pattern = rf"{label}\s*[:\-]?\s*(.+)"
        m = re.search(pattern, text, re.IGNORECASE)
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
        r"net\s*amount",
        r"amount(?:\s*before\s*tax)?",
    ])
    if subtotal_val:
        m = re.search(NUM_RE, subtotal_val)
        if m:
            subtotal = clean_number(m.group(0))

    tax_val = find_label_value(text, [
        r"gst\s*\([\d.]+%\)",
        r"gst",
        r"vat\s*\([\d.]+%\)",
        r"vat",
        r"tax(?:\s*\([\d.]+%\))?",
        r"sales\s*tax",
    ])
    if tax_val:
        m = re.search(NUM_RE, tax_val)
        if m:
            tax = clean_number(m.group(0))

    total_val = find_label_value(text, [
        r"total",
        r"grand\s*total",
    ])
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
