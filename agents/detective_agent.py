import random

class DetectiveAgent:
    def analyze_network(self, claim_id):
        # In a real scenario, this would load the Graph Data and run the GNN model inference
        # For demo purposes, we simulate a high-risk finding
        risk_score = 0.85
        reasoning = "Claimant shares a phone number with 2 previously flagged fraudulent IDs."
        return risk_score, reasoning
