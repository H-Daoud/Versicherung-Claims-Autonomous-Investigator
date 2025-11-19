import os

class JudgeAgent:
    def __init__(self):
        # Check if API Key exists
        self.api_key = os.getenv("OPENAI_API_KEY")
        
    def decide(self, claim_text, risk_score, fraud_reasoning, coverage_analysis):
        print(f"   [Judge] Synthesizing evidence...")

        # The System Prompt (Agentic Persona)
        system_prompt = """
        You are a Senior Insurance Adjudicator for HUK-Coburg. 
        Your job is to make a final claim decision based on evidence from two sub-agents:
        1. The Detective (Fraud Analysis)
        2. The Lawyer (Policy Coverage)
        
        Be decisive, objective, and cite the specific evidence provided.
        """
        
        # The User Prompt (The specific case)
        user_prompt = f"""
        --- CASE DATA ---
        CLAIM: "{claim_text}"
        
        EVIDENCE A (Detective / Fraud GNN):
        - Risk Score: {risk_score}/1.0
        - Finding: {fraud_reasoning}
        
        EVIDENCE B (Lawyer / Policy RAG):
        - Policy Analysis: {coverage_analysis}
        
        --- MISSION ---
        1. Determine the Final Verdict (APPROVE, REJECT, or INVESTIGATE FURTHER).
        2. Write a formal explanation for the internal file.
        """

        # --- LLM EXECUTION ---
        if self.api_key:
            # If you have a key, this runs for real
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        else:
            # --- SIMULATION MODE (For GitHub/Recruiters without keys) ---
            return self._simulated_response(risk_score, coverage_analysis)

    def _simulated_response(self, risk_score, coverage_analysis):
        # This ensures the code runs even if the user doesn't have an API key
        if risk_score > 0.8:
            verdict = "REJECT"
            reason = "High risk of organized fraud network detected."
        elif "VOID" in coverage_analysis:
            verdict = "REJECT"
            reason = "Policy exclusion applies (Commercial Use)."
        else:
            verdict = "APPROVE"
            reason = "Claim falls within coverage limits and low fraud risk."

        return f"""
        *** SIMULATED LLM OUTPUT (No API Key Detected) ***
        
        FINAL VERDICT: {verdict}
        
        REASONING:
        Based on the Detective's finding of a high risk score ({risk_score}) and the 
        identified connection to a known fraud ring, the system recommends rejection 
        regardless of policy coverage.
        
        {reason}
        """
