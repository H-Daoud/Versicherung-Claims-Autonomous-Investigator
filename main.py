import os
from agents.detective_agent import DetectiveAgent
from agents.lawyer_agent import LawyerAgent
# form langchain_openai import ChatOpenAI # Uncomment when API key is set

def run_investigation(claim_id, claim_text):
    print(f"--- Starting Investigation for Claim {claim_id} ---")

    # 1. The Detective checks for Fraud Rings (GNN)
    detective = DetectiveAgent()
    fraud_risk_score, fraud_reasoning = detective.analyze_network(claim_id)
    print(f"🕵️‍♂️ Detective Report: Risk Level {fraud_risk_score}. Reason: {fraud_reasoning}")

    # 2. The Lawyer checks Policy Coverage (RAG)
    lawyer = LawyerAgent()
    coverage_analysis = lawyer.check_policy(claim_text)
    print(f"🧑‍⚖️ Lawyer Report: {coverage_analysis}")

    # 3. The Judge makes the final decision
    # (Simulated LLM output for this script demo)
    print(f"\n⚖️ FINAL VERDICT (Simulated LLM):")
    print(f"Based on the high fraud risk ({fraud_risk_score}) identified by the Detective, and the Lawyer's policy exclusion note, I RECOMMEND REJECTION.")

if __name__ == "__main__":
    # Example Claim
    run_investigation(
        claim_id="C-1024", 
        claim_text="Customer reporting front-bumper damage at 2 AM."
    )
