"""
Terminal CLI client for the Lead Score Agent.

Run the API server first (in another terminal):
    uvicorn main:app --reload --port 8000

Then run this client:
    python client.py            -> interactive prompt mode
    python client.py --history  -> show all past scored leads
"""

import sys
import requests

API_URL = "http://127.0.0.1:8000"

INCOME_SOURCES = [
    "salaried_govt",
    "salaried_mnc",
    "salaried_private",
    "self_employed_professional",
    "business_owner",
    "freelance",
    "unemployed",
]

LOAN_HISTORY_OPTIONS = [
    "no_history",
    "all_paid_on_time",
    "minor_delays",
    "one_default",
    "multiple_defaults",
]


def ask(prompt, cast=str, choices=None):
    while True:
        raw = input(prompt).strip()
        if choices and raw.lower() not in choices:
            print(f"  -> please choose one of: {choices}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  -> invalid value, try again")


def print_choices(label, options):
    print(f"{label}: " + ", ".join(options))


def run_interactive():
    print("=" * 60)
    print(" LEAD SCORE AGENT - New Lead Intake")
    print("=" * 60)

    lead_id = input("Lead ID (optional, press enter to skip): ").strip() or None
    name = input("Customer Name (optional): ").strip() or None
    source = input("Lead Source [CRM/Website/App] (default 'manual'): ").strip() or "manual"

    cibil_score = ask("CIBIL Score (300-900): ", int)
    annual_income = ask("Annual Income (INR): ", float)
    assets_value = ask("Assets Value (INR, 0 if none): ", float)

    print_choices("Income Source options", INCOME_SOURCES)
    income_source = ask("Income Source: ", str, choices=INCOME_SOURCES)

    print_choices("Previous Loan History options", LOAN_HISTORY_OPTIONS)
    previous_loan_history = ask("Previous Loan History: ", str, choices=LOAN_HISTORY_OPTIONS)

    payload = {
        "lead_id": lead_id,
        "name": name,
        "source": source,
        "cibil_score": cibil_score,
        "annual_income": annual_income,
        "assets_value": assets_value,
        "income_source": income_source,
        "previous_loan_history": previous_loan_history,
    }

    try:
        resp = requests.post(f"{API_URL}/score", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not reach the agent server.")
        print("Start it first with: uvicorn main:app --reload --port 8000")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"\nERROR ({resp.status_code}): {resp.text}")
        return

    result = resp.json()
    print_result(result)


def print_result(result):
    print("\n" + "=" * 60)
    print(" SCORING RESULT")
    print("=" * 60)
    print(f" Record ID      : {result.get('record_id')}")
    print(f" Lead ID        : {result.get('lead_id')}")
    print(f" Name           : {result.get('name')}")
    print(f" Total Score    : {result.get('total_score')}/100")
    print(f" Category       : {result.get('category')}")
    print(f" Recommendation : {result.get('recommendation')}")
    print(" Breakdown      :")
    for k, v in result.get("breakdown", {}).items():
        print(f"    - {k}: {v}")
    print(" Reason Codes   :")
    for r in result.get("reason_codes", []):
        print(f"    - {r}")
    print(f" Timestamp      : {result.get('timestamp')}")
    print("=" * 60 + "\n")


def show_history():
    try:
        resp = requests.get(f"{API_URL}/leads", timeout=10)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not reach the agent server. Start it with uvicorn first.")
        sys.exit(1)

    leads = resp.json()
    if not leads:
        print("No leads scored yet.")
        return

    print(f"{'ID':<4}{'Name':<15}{'Score':<8}{'Category':<16}{'Recommendation'}")
    print("-" * 70)
    for l in leads:
        print(
            f"{l.get('record_id', ''):<4}"
            f"{(l.get('name') or '-')[:14]:<15}"
            f"{l.get('total_score'):<8}"
            f"{l.get('category'):<16}"
            f"{l.get('recommendation')}"
        )


if __name__ == "__main__":
    if "--history" in sys.argv:
        show_history()
    else:
        run_interactive()