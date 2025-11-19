import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from models.gnn_fraud_detector import FraudGNN

class DetectiveAgent:
    def __init__(self):
        # Initialize the GNN Model
        # Input dim: 10 features, Hidden: 16, Output: 2 (Fraud vs Legit)
        self.model = FraudGNN(input_dim=10, hidden_dim=16, output_dim=2)
        self.model.eval()  # Set to evaluation mode

    def _build_claim_graph(self, claim_id):
        """
        Simulates fetching data from a GraphDB (like Neo4j) and building a 
        local subgraph for analysis.
        """
        G = nx.Graph()
        
        # Create the central node (The Claim)
        G.add_node(claim_id, x=np.random.rand(10)) # 10 random features
        
        # Create related nodes (The Network)
        # In a real scenario, these are fetched based on shared phone numbers, IPs, etc.
        G.add_node("Person_A", x=np.random.rand(10)) # The Claimant
        G.add_node("Shop_X", x=np.random.rand(10))   # The Repair Shop
        G.add_node("Person_B", x=np.random.rand(10)) # A known fraudster
        
        # Connect them
        G.add_edge(claim_id, "Person_A")
        G.add_edge(claim_id, "Shop_X")
        
        # THE RED FLAG: The Repair Shop is connected to a known fraudster
        G.add_edge("Shop_X", "Person_B")
        
        return G

    def analyze_network(self, claim_id):
        print(f"   [Detective] Building Knowledge Graph for {claim_id}...")
        
        # 1. Build the graph
        nx_graph = self._build_claim_graph(claim_id)
        
        # 2. Convert to PyTorch Geometric Data
        # We need to manually collate node features 'x' into a tensor
        features = [nx_graph.nodes[n]['x'] for n in nx_graph.nodes]
        x = torch.tensor(np.array(features), dtype=torch.float)
        
        # Convert graph structure
        pyg_data = from_networkx(nx_graph)
        pyg_data.x = x # Attach features
        
        # 3. Run Inference
        with torch.no_grad():
            # For demo: We pass dummy edge indices. 
            # In production, 'from_networkx' handles this.
            logits = self.model(pyg_data.x, pyg_data.edge_index)
            probs = torch.exp(logits)
        
        # 4. Interpret Results
        # (For this demo, we force a high risk score to show the workflow)
        # In a real training loop, the GNN would learn this from data.
        risk_score = 0.89 
        
        reasoning = (
            "Graph Analysis detected a 2-hop connection to a flagged entity ('Person_B') "
            "via the repair shop node. This matches known fraud ring patterns."
        )
        
        return risk_score, reasoning
