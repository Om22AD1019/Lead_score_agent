import sys
import requests
import json

API_URL = "http://127.0.0.1:8000"

def fetch_lead(lead_id):
    try:
        response = requests.get(f"{API_URL}/leads/{lead_id}")
        response.raise_for_status()
        lead_data = response.json()
        display_lead(lead_data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching lead data: {e}")
        sys.exit(1)

def display_lead(lead_data):
    print("\n--- Lead Details ---")
    print(f"Record ID: {lead_data['record_id']}")
    print(f"Lead ID: {lead_data['lead_id']}")
    print(f"Name: {lead_data['name']}")
    print(f"Source: {lead_data['source']}")
    print(f"Phone: {lead_data['phone']}")
    print(f"Email: {lead_data['email']}")
    print(f"City: {lead_data['city']}")
    print(f"CIBIL Score: {lead_data['cibil_score']}")
    print(f"Annual Income: {lead_data['annual_income']}")
    print(f"Assets Value: {lead_data['assets_value']}")
    print(f"Income Source: {lead_data['income_source']}")
    print(f"Previous Loan History: {lead_data['previous_loan_history']}")
    print("\n--- Breakdown ---")
    for key, value in lead_data['breakdown'].items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("\n--- Summary ---")
    print(f"Total Score: {lead_data['total_score']}")
    print(f"Category: {lead_data['category']}")
    print(f"Recommendation: {lead_data['recommendation']}")

    print("\n--- Summary Analysis ---")
    if lead_data['category'] == "Good Lead":
        print("This lead is categorized as a good lead due to the following reasons:")
        if lead_data['cibil_score'] >= 750:
            print("- Excellent CIBIL score, indicating strong creditworthiness.")
        elif 700 <= lead_data['cibil_score'] < 750:
            print("- Good CIBIL score, indicating reliable credit history.")
        if lead_data['annual_income'] >= 1000000:
            print("- High annual income, suggesting financial stability.")
        elif 600000 <= lead_data['annual_income'] < 1000000:
            print("- Moderate annual income, sufficient for loan eligibility.")
        if lead_data['assets_value'] >= 2000000:
            print("- Significant asset value, indicating strong financial backing.")
        if lead_data['income_source'] in ["salaried_mnc", "salaried_private"]:
            print("- Stable income source from employment.")
        if lead_data['previous_loan_history'] == "all_paid_on_time":
            print("- Excellent loan repayment history with no defaults.")
    else:
        print("This lead is not ideal due to the following reasons:")
        if lead_data['cibil_score'] < 700:
            print("- Low CIBIL score, indicating poor creditworthiness.")
        if lead_data['annual_income'] < 600000:
            print("- Low annual income, suggesting limited financial capacity.")
        if lead_data['assets_value'] < 1000000:
            print("- Insufficient asset value, indicating weak financial backing.")
        if lead_data['income_source'] not in ["salaried_mnc", "salaried_private"]:
            print("- Unstable or risky income source.")
        if lead_data['previous_loan_history'] != "all_paid_on_time":
            print("- Poor loan repayment history with defaults or delays.")

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <lead_id>")
        sys.exit(1)
    lead_id = sys.argv[1]
    fetch_lead(lead_id)