"""
Lead Score Agent — Terminal Client
Powered by Groq AI (Llama 3)

USAGE:
  python client.py <record_id>   -> fetch & display a lead
  python client.py --history     -> view all leads as a table
  python client.py --score       -> manual intake (fill fields one by one)
  python client.py --smart       -> smart intake (describe in plain English)
  python client.py --chat        -> chat with your leads data
  python client.py --help        -> show help
"""

import sys
import json
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()   # reads from .env file

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Config ────────────────────────────────────────────────────
API_URL    = os.getenv("API_URL", "http://127.0.0.1:8000")
GROQ_KEY   = os.getenv("GROQ_API_KEY")   # reads from .env
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "groq/compound-mini"
INCOME_SOURCES = [
    "salaried_govt", "salaried_mnc", "salaried_private",
    "self_employed_professional", "business_owner", "freelance", "unemployed",
]
LOAN_HISTORY_OPTIONS = [
    "no_history", "all_paid_on_time", "minor_delays",
    "one_default", "multiple_defaults",
]
CATEGORY_ICONS = {
    "Excellent Lead": "★★★★",
    "Good Lead":      "★★★☆",
    "Average Lead":   "★★☆☆",
    "Poor Lead":      "★☆☆☆",
}


# ══════════════════════════════════════════════════════════════
#  GEMINI AI CALLER
# ══════════════════════════════════════════════════════════════

def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """
    Call Groq API (OpenAI-compatible).
    Auto-retries on rate limit: waits 10s → 20s → 30s.
    Groq free tier: 30 requests/minute, very fast inference.
    """
    RETRIES    = 3
    WAIT_TIMES = [10, 20, 30]

    for attempt in range(RETRIES):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.5,
                },
                timeout=30,
            )

            # ── Success ───────────────────────────────────────
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    return f"[Unexpected response: {str(data)[:200]}]"

            # ── Rate limit → auto retry ───────────────────────
            elif resp.status_code == 429:
                wait = WAIT_TIMES[attempt]
                if attempt < RETRIES - 1:
                    print(f"  [Rate limit] Waiting {wait}s, retrying "
                          f"({attempt + 1}/{RETRIES})...   ", end="\r")
                    time.sleep(wait)
                    continue
                else:
                    return (
                        "[Rate limit reached after 3 retries. "
                        "Groq free tier: 30 req/min. Wait a moment and try again.]"
                    )

            # ── Other errors ──────────────────────────────────
            elif resp.status_code == 401:
                return "[Invalid Groq API key — check your key]"
            elif resp.status_code == 400:
                msg = resp.json().get("error", {}).get("message", resp.text[:150])
                return f"[Bad request: {msg}]"
            else:
                return f"[Groq API Error {resp.status_code}: {resp.text[:150]}]"

        except requests.exceptions.ConnectionError:
            return "[Network error — check your internet connection]"
        except requests.exceptions.Timeout:
            if attempt < RETRIES - 1:
                wait = WAIT_TIMES[attempt]
                print(f"  [Timeout] Retrying in {wait}s...", end="\r")
                time.sleep(wait)
                continue
            return "[Timed out after 3 retries]"
        except Exception as e:
            return f"[Error: {e}]"

    return "[Failed after all retries]"


# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def score_bar(score, width=20):
    filled = int((score / 100) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score}/100"

def mini_bar(got, max_marks, width=10):
    filled = int((got / max_marks) * width) if max_marks else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"

def wrap_text(text, width=62, indent="  "):
    words = text.split()
    lines, line = [], []
    for w in words:
        if len(" ".join(line + [w])) > width:
            lines.append(indent + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)

def server_get(path):
    try:
        return requests.get(f"{API_URL}{path}", timeout=10)
    except requests.exceptions.ConnectionError:
        print("\n  ERROR: Cannot reach the server.")
        print("  Start it with:  uvicorn main:app --reload --port 8000")
        sys.exit(1)

def ask_input(prompt, cast=str, choices=None):
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  -> value required")
            continue
        if choices and raw.lower() not in [c.lower() for c in choices]:
            print(f"  -> choose one of: {', '.join(choices)}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  -> invalid value, try again")


# ══════════════════════════════════════════════════════════════
#  FETCH LEAD
# ══════════════════════════════════════════════════════════════

def fetch_lead(lead_id: str):
    """Fetch by record_id (number) or lead_id string (LEAD-xxx)."""
    if lead_id.isdigit():
        try:
            response = requests.get(f"{API_URL}/leads/{lead_id}", timeout=10)
            response.raise_for_status()
            display_lead(response.json())
        except requests.exceptions.HTTPError:
            print(f"\n  No lead found with record ID: {lead_id}")
        except requests.exceptions.RequestException as e:
            print(f"\n  Error fetching lead: {e}")
            sys.exit(1)
    else:
        resp    = server_get("/leads")
        matched = [l for l in resp.json()
                   if l.get("lead_id", "").upper() == lead_id.upper()]
        if not matched:
            print(f"\n  No lead found with Lead ID: '{lead_id}'")
            return
        display_lead(matched[0])

# save_summary_to_json function to save AI summary to ai_summaries.json#

def save_summary_to_json(lead_id, summary):
    file_path = "ai_summaries.json"

    # Check if the file exists, if not, create it with an empty list
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=4)

    # Check if the file is empty
    if os.path.getsize(file_path) == 0:
        summaries = []  # Initialize an empty list if the file is empty
    else:
        # Load existing summaries with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as file:
            summaries = json.load(file)

    # Append the new summary
    summaries.append({"lead_id": lead_id, "summary": summary})

    # Write back to the file with UTF-8 encoding
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=4)

    print(f"Summary for Lead ID {lead_id} saved to {file_path}")

# ══════════════════════════════════════════════════════════════
#  DISPLAY LEAD
# ══════════════════════════════════════════════════════════════

def display_lead(lead_data: dict):
    icon    = CATEGORY_ICONS.get(lead_data.get("category", ""), "")
    divider = "=" * 65

    print("\n" + divider)
    print(f"  LEAD DETAILS  {icon}  {lead_data.get('category', '').upper()}")
    print(divider)

    # ── Customer Info ─────────────────────────────────────────
    print(f"\n  {'Record ID':<24}: {lead_data.get('record_id')}")
    print(f"  {'Lead ID':<24}: {lead_data.get('lead_id')}")
    print(f"  {'Name':<24}: {lead_data.get('name')}")
    print(f"  {'Source':<24}: {lead_data.get('source')}")
    print(f"  {'Phone':<24}: {lead_data.get('phone') or 'N/A'}")
    print(f"  {'Email':<24}: {lead_data.get('email') or 'N/A'}")
    print(f"  {'City':<24}: {lead_data.get('city') or 'N/A'}")
    print(f"  {'CIBIL Score':<24}: {lead_data.get('cibil_score')}")
    print(f"  {'Annual Income':<24}: INR {lead_data.get('annual_income', 0):>12,.0f}")
    print(f"  {'Assets Value':<24}: INR {lead_data.get('assets_value', 0):>12,.0f}")
    print(f"  {'Income Source':<24}: {lead_data.get('income_source')}")
    print(f"  {'Previous Loan History':<24}: {lead_data.get('previous_loan_history')}")

    # ── Breakdown ─────────────────────────────────────────────
    print(f"\n--- Breakdown ---")
    bd = lead_data.get("breakdown", {})
    labels = {
        "cibil_score_marks":    ("CIBIL Score",   35),
        "annual_income_marks":  ("Annual Income", 25),
        "assets_value_marks":   ("Assets Value",  15),
        "income_source_marks":  ("Income Source", 10),
        "previous_loans_marks": ("Loan History",  15),
    }
    for key, (label, max_marks) in labels.items():
        got = bd.get(key, 0)
        print(f"  {label:<22}: {got:>2}/{max_marks}  {mini_bar(got, max_marks)}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n--- Summary ---")
    total = lead_data.get("total_score", 0)
    print(f"  Total Score    : {score_bar(total)}")
    print(f"  Category       : {lead_data.get('category')}  {icon}")
    print(f"  Recommendation : {lead_data.get('recommendation')}")

    # ── Reason Codes ──────────────────────────────────────────
    print(f"\n--- Reason Codes ---")
    for r in lead_data.get("reason_codes", []):
        print(f"  ✦  {r}")

    # ── Rule-based Analysis ───────────────────────────────────
    print(f"\n--- Summary Analysis ---")
    print_rule_based_summary(lead_data)

    # ── Gemini AI Summary ─────────────────────────────────────
    print(f"\n--- AI Summary (Powered by Groq / Llama 3) ---")
    print("  Generating...", end="\r")
    ai_text = generate_ai_summary(lead_data)
    print("                \r" + wrap_text(ai_text))

    # Save the AI summary to ai_summaries.json
    save_summary_to_json(lead_data.get("lead_id"), ai_text)

    print("\n" + divider + "\n")


def print_rule_based_summary(lead: dict):
    cibil   = lead.get("cibil_score", 0)
    income  = lead.get("annual_income", 0)
    assets  = lead.get("assets_value", 0)
    src     = lead.get("income_source", "")
    history = lead.get("previous_loan_history", "")

    positives, negatives = [], []

    # CIBIL
    if cibil >= 750:
        positives.append(f"Excellent CIBIL score ({cibil}) — strong creditworthiness")
    elif cibil >= 700:
        positives.append(f"Good CIBIL score ({cibil}) — reliable credit history")
    elif cibil >= 650:
        negatives.append(f"Fair CIBIL score ({cibil}) — borderline creditworthiness")
    else:
        negatives.append(f"Low CIBIL score ({cibil}) — poor creditworthiness")

    # Income
    if income >= 1_500_000:
        positives.append(f"High annual income (INR {income:,.0f}) — strong repayment capacity")
    elif income >= 1_000_000:
        positives.append(f"Good annual income (INR {income:,.0f}) — financially stable")
    elif income >= 600_000:
        positives.append(f"Moderate income (INR {income:,.0f}) — meets eligibility")
    else:
        negatives.append(f"Low income (INR {income:,.0f}) — limited repayment capacity")

    # Assets
    if assets >= 5_000_000:
        positives.append(f"Strong assets (INR {assets:,.0f}) — excellent financial backing")
    elif assets >= 2_000_000:
        positives.append(f"Good assets (INR {assets:,.0f}) — solid collateral potential")
    elif assets >= 500_000:
        positives.append(f"Moderate assets (INR {assets:,.0f})")
    elif assets > 0:
        negatives.append(f"Weak assets (INR {assets:,.0f}) — limited backing")
    else:
        negatives.append("No declared assets")

    # Income source
    src_map = {
        "salaried_govt":              ("Government employment — most stable", True),
        "salaried_mnc":               ("MNC/Corporate employment — very stable", True),
        "salaried_private":           ("Private company employment — stable", True),
        "self_employed_professional": ("Self-employed professional — moderate stability", True),
        "business_owner":             ("Business owner — variable income", False),
        "freelance":                  ("Freelance/gig income — irregular", False),
        "unemployed":                 ("No income source — high risk", False),
    }
    if src in src_map:
        msg, is_good = src_map[src]
        (positives if is_good else negatives).append(msg)

    # Loan history
    hist_map = {
        "all_paid_on_time":   ("Perfect loan repayment history", True),
        "no_history":         ("No prior loan history — neutral", None),
        "minor_delays":       ("Minor repayment delays on record", False),
        "one_default":        ("One loan default on record", False),
        "multiple_defaults":  ("Multiple loan defaults — very risky", False),
    }
    if history in hist_map:
        msg, is_good = hist_map[history]
        if is_good is True:
            positives.append(msg)
        elif is_good is False:
            negatives.append(msg)
        else:
            print(f"  ○  {msg}")

    if positives:
        print("  Strengths:")
        for p in positives:
            print(f"    ✔  {p}")
    if negatives:
        print("  Weaknesses:")
        for n in negatives:
            print(f"    ✘  {n}")


# ══════════════════════════════════════════════════════════════
#  AI SUMMARY — Gemini
# ══════════════════════════════════════════════════════════════

def generate_ai_summary(lead: dict) -> str:
    prompt = (
        f"You are a loan lead analyst. Write a 3-sentence plain-English summary "
        f"of this customer for the sales team. Mention actual numbers. "
        f"End with a clear action — should the team pursue this lead or not?\n\n"
        f"Name: {lead.get('name')} | CIBIL: {lead.get('cibil_score')} | "
        f"Income: INR {lead.get('annual_income', 0):,} | "
        f"Assets: INR {lead.get('assets_value', 0):,} | "
        f"Source: {lead.get('income_source')} | "
        f"Loan History: {lead.get('previous_loan_history')} | "
        f"Score: {lead.get('total_score')}/100 | "
        f"Category: {lead.get('category')} | "
        f"Recommendation: {lead.get('recommendation')}\n\n"
        f"Write ONLY the 3-sentence summary. No bullet points, no headings."
    )
    return call_llm(prompt, max_tokens=150)


# ══════════════════════════════════════════════════════════════
#  SMART INTAKE — plain English → structured lead via Gemini
# ══════════════════════════════════════════════════════════════

def smart_intake():
    print("\n" + "=" * 65)
    print("  SMART INTAKE — Describe the customer in plain English")
    print("=" * 65)
    print("  Example: 'Priya is a govt teacher in Chennai,")
    print("  earns 9 lakhs a year, CIBIL 740, owns property")
    print("  worth 25 lakhs, never defaulted on any loan.'")
    print("-" * 65)

    description = input("\n  Your description:\n  > ").strip()
    if not description:
        print("  No input provided.")
        return

    print("\n  Extracting fields via Gemini...")

    prompt = (
        "Extract loan lead data from this customer description and return ONLY a valid JSON object.\n"
        "Required keys: lead_id, name, source, phone, email, city, "
        "cibil_score (int 300-900), annual_income (float INR, 1 lakh = 100000), "
        "assets_value (float INR), income_source, previous_loan_history.\n\n"
        "income_source must be one of: salaried_govt, salaried_mnc, salaried_private, "
        "self_employed_professional, business_owner, freelance, unemployed\n"
        "previous_loan_history must be one of: no_history, all_paid_on_time, minor_delays, "
        "one_default, multiple_defaults\n\n"
        "Defaults if not mentioned: lead_id=null, phone=null, email=null, "
        "source='manual', city=null, assets_value=0, previous_loan_history='no_history'\n\n"
        f"Description: {description}\n\n"
        "Return ONLY the JSON object. No explanation, no markdown, no code fences."
    )

    raw = call_llm(prompt, max_tokens=300)

    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        extracted = json.loads(clean[start:end])
    except Exception:
        print(f"\n  Could not parse Gemini output:\n  {raw}")
        print("\n  Falling back to manual intake...")
        manual_score()
        return

    print("\n  ── Extracted Fields ─────────────────────────────────")
    for k, v in extracted.items():
        if v is not None:
            print(f"  {k:<28}: {v}")

    confirm = input("\n  Looks correct? Submit for scoring? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    try:
        resp = requests.post(f"{API_URL}/score", json=extracted, timeout=10)
    except requests.exceptions.ConnectionError:
        print("\n  ERROR: Cannot reach the server.")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"\n  ERROR ({resp.status_code}): {resp.text}")
        return

    display_lead(resp.json())


# ══════════════════════════════════════════════════════════════
#  CHAT MODE — multi-turn conversation about your leads
# ══════════════════════════════════════════════════════════════

def chat_mode():
    print("\n" + "=" * 65)
    print("  CHAT MODE — Ask anything about your leads (Groq AI)")
    print("=" * 65)
    print("  Try:")
    print("    'Who has the highest score?'")
    print("    'Show all excellent leads'")
    print("    'Which Mumbai leads should I call first?'")
    print("    'How many poor leads do we have?'")
    print("    'Compare LEAD-005 and LEAD-010'")
    print("  Type 'exit' to quit.")
    print("-" * 65)

    resp      = server_get("/leads")
    all_leads = resp.json()

    if not all_leads:
        print("  No leads in the system yet.")
        return

    # Compact context of all leads for Gemini
    leads_context = "\n".join([
        f"ID:{l.get('record_id')} LeadID:{l.get('lead_id','-')} "
        f"Name:{l.get('name','-')} City:{l.get('city','-')} "
        f"Score:{l.get('total_score')} Category:{l.get('category')} "
        f"CIBIL:{l.get('cibil_score')} Income:{l.get('annual_income')} "
        f"Assets:{l.get('assets_value')} Source:{l.get('income_source')} "
        f"History:{l.get('previous_loan_history')} Action:{l.get('recommendation')}"
        for l in all_leads
    ])

    conversation_history = []

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting chat.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("  Exiting chat.")
            break
        if not user_input:
            continue

        # Build prompt with last 4 turns for context
        history_text = ""
        for turn in conversation_history[-4:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt = (
            f"You are a loan lead analyst assistant. Answer questions about the lead database below.\n"
            f"Use plain text only. No markdown, no bullet points with asterisks.\n\n"
            f"LEADS DATABASE:\n{leads_context}\n\n"
            f"{history_text}"
            f"User: {user_input}"
        )

        print("  Agent: ", end="", flush=True)
        reply = call_llm(prompt, max_tokens=400)
        print(reply)

        conversation_history.append({"user": user_input, "assistant": reply})


# ══════════════════════════════════════════════════════════════
#  MANUAL SCORE
# ══════════════════════════════════════════════════════════════

def manual_score():
    print("\n" + "=" * 65)
    print("  MANUAL INTAKE — Fill fields one by one")
    print("=" * 65)

    lead_id = input("  Lead ID (optional): ").strip() or None
    name    = input("  Customer Name: ").strip() or None
    phone   = input("  Phone (optional): ").strip() or None
    email   = input("  Email (optional): ").strip() or None
    city    = input("  City (optional): ").strip() or None
    source  = input("  Source [CRM/Website/App] (default: manual): ").strip() or "manual"

    cibil_score   = ask_input("  CIBIL Score (300-900): ", int)
    annual_income = ask_input("  Annual Income (INR): ", float)
    assets_value  = ask_input("  Assets Value (INR, 0 if none): ", float)

    print(f"\n  Income Sources: {', '.join(INCOME_SOURCES)}")
    income_source = ask_input("  Income Source: ", str, choices=INCOME_SOURCES)

    print(f"\n  Loan History: {', '.join(LOAN_HISTORY_OPTIONS)}")
    previous_loan_history = ask_input("  Loan History: ", str, choices=LOAN_HISTORY_OPTIONS)

    payload = {
        "lead_id": lead_id, "name": name, "source": source,
        "phone": phone, "email": email, "city": city,
        "cibil_score": cibil_score, "annual_income": annual_income,
        "assets_value": assets_value, "income_source": income_source,
        "previous_loan_history": previous_loan_history,
    }

    try:
        resp = requests.post(f"{API_URL}/score", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        print("\n  ERROR: Cannot reach the server.")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"\n  ERROR ({resp.status_code}): {resp.text}")
        return

    display_lead(resp.json())


# ══════════════════════════════════════════════════════════════
#  HISTORY TABLE
# ══════════════════════════════════════════════════════════════

def show_history():
    resp  = server_get("/leads")
    leads = resp.json()
    if not leads:
        print("  No leads found.")
        return

    print("\n" + "=" * 78)
    print(f"  {'ID':<5} {'Lead ID':<12} {'Name':<18} {'Score':<7} {'Category':<20} Recommendation")
    print("-" * 78)
    for l in leads:
        cat  = l.get("category", "")
        icon = CATEGORY_ICONS.get(cat, "")
        print(
            f"  {str(l.get('record_id','')):<5}"
            f"{(l.get('lead_id') or '-'):<12}"
            f"{(l.get('name') or '-')[:17]:<18}"
            f"{str(l.get('total_score')):<7}"
            f"{(cat + ' ' + icon)[:22]:<24}"
            f"{l.get('recommendation','')}"
        )
    print("=" * 78 + "\n")


# ══════════════════════════════════════════════════════════════
#  HELP
# ══════════════════════════════════════════════════════════════

def print_help():
    print("""
  Lead Score Agent — Commands
  ─────────────────────────────────────────────────────────────
  python client.py <id>          Fetch lead (record ID or Lead ID)
  python client.py --score       Manual intake (fill fields one by one)
  python client.py --smart       Smart intake (describe in plain English)
  python client.py --history     View all leads as a table
  python client.py --chat        Chat with your leads data (Groq AI)
  python client.py --help        Show this help

  Examples:
    python client.py 15
    python client.py LEAD-015
    python client.py --chat
    python client.py --smart
""")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print_help()
    elif args[0] == "--history":
        show_history()
    elif args[0] == "--score":
        manual_score()
    elif args[0] == "--smart":
        smart_intake()
    elif args[0] == "--chat":
        chat_mode()
    elif args[0] == "--help":
        print_help()
    else:
        fetch_lead(args[0])