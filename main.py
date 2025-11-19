from agents.detective_agent import DetectiveAgent
from agents.lawyer_agent import LawyerAgent
from agents.judge_agent import JudgeAgent

def run_investigation(claim_id, claim_text):
    print(f"\n🚀 --- STARTING AUTONOMOUS INVESTIGATION: {claim_id} ---")
    print(f"📝 Claim Description: \"{claim_text}\"\n")

    # 1. The Detective (GNN Analysis)
    detective = DetectiveAgent()
    fraud_risk_score, fraud_reasoning = detective.analyze_network(claim_id)
    print(f"🔎 [Detective Report]: Risk Score {fraud_risk_score}")
    print(f"   -> {fraud_reasoning}\n")

    # 2. The Lawyer (Policy/RAG Check)
    lawyer = LawyerAgent()
    coverage_analysis = lawyer.check_policy(claim_text)
    print(f"⚖️ [Lawyer Report]: Policy Check Complete")
    print(f"   -> {coverage_analysis}\n")

    # 3. The Judge (Final Decision)
    judge = JudgeAgent()
    final_verdict = judge.decide(claim_text, fraud_risk_score, fraud_reasoning, coverage_analysis)
    
    print(f"👨‍⚖️ [FINAL VERDICT]:")
    print("-" * 40)
    print(final_verdict)
    print("-" * 40)

if __name__ == "__main__":
    # Scenario: A claim that looks structurally suspicious (fraud ring) 
    # AND has a policy issue (delivery/commercial use).
    run_investigation(
        claim_id="CLM-2025-88X", 
        claim_text="I was doing a food delivery when I hit a bumper at 2 AM near the warehouse."
    )