class LawyerAgent:
    def __init__(self, policy_path="data/policy_docs.txt"):
        self.policy_path = policy_path
        self.policy_text = self._load_policy()

    def _load_policy(self):
        try:
            with open(self.policy_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "Error: Policy documents not found."

    def check_policy(self, claim_text):
        print(f"   [Lawyer] Retrieving relevant policy clauses...")
        
        # --- RAG SIMULATION ---
        # In a full production system, we would use:
        # 1. Vector Store (Pinecone) -> vector_store.similarity_search(claim_text)
        # 2. Retrieve top 3 chunks
        
        # For this demo, we use keyword logic to simulate Retrieval
        relevant_clauses = []
        
        if "commercial" in claim_text.lower() or "delivery" in claim_text.lower():
            relevant_clauses.append("Clause 2.1: Coverage VOID for unregistered commercial use.")
        
        if "collision" in claim_text.lower() or "bumper" in claim_text.lower():
            relevant_clauses.append("Clause 1.1: Collision coverage applies.")
            
        if not relevant_clauses:
            return "No specific exclusions found. Standard coverage likely applies."
            
        # Return the findings
        return f"Policy Scan Results: {'; '.join(relevant_clauses)}"
