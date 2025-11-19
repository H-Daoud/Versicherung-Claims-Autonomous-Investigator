# 🕵️‍♂️ Autonomous Claims Investigator: Agentic AI + Graph Neural Networks

## 🚀 Project Overview
This project demonstrates a **Compound AI System** designed to automate complex insurance claim investigations. Instead of a simple "black box" prediction, it orchestrates three specialized AI agents to reason about fraud, policy coverage, and causality.

## 🏗️ Architecture
The system utilizes a Multi-Agent workflow:
1.  **The Detective (Graph Neural Network):** Analyzes the network structure (Claims ↔ People ↔ Repair Shops) to detect hidden fraud rings using **PyTorch Geometric**.
2.  **The Lawyer (RAG - Retrieval Augmented Generation):** Queries insurance policy documents to verify coverage logic using **LangChain**.
3.  **The Judge (LLM Orchestrator):** Synthesizes the evidence and makes a final, explained decision.

## 🛠️ Tech Stack
* **Graph ML:** PyTorch Geometric (GNNs)
* **Orchestration:** LangGraph / LangChain
* **LLM:** OpenAI GPT-4o (or Local Llama 3)
* **Data:** NetworkX (Synthetic Knowledge Graph generation)

## 📉 Business Impact
* **Fraud Detection:** Detects organized fraud rings that standard tabular models miss.
* **Explainability:** Provides a "Chain of Thought" reasoning, not just a probability score.

## Project Structure
```bash
npm install
npm run dev

claims-autonomous-investigator/
│
├── data/
│   ├── synthetic_graph_data.py    # Generates a fake Knowledge Graph (Claims, People, Shops)
│   └── policy_docs.txt            # Mock insurance policy text for RAG
│
├── models/
│   └── gnn_fraud_detector.py      # PyTorch Geometric model (The "Detective")
│
├── agents/
│   ├── detective_agent.py         # Wraps the GNN model
│   ├── lawyer_agent.py            # Handles RAG / Policy lookup
│   └── judge_agent.py             # LangGraph Orchestrator (The final decision)
│
├── notebooks/
│   └── prototype_demo.ipynb       # Jupyter notebook for visual demo
│
├── main.py                        # Main entry point to run the system
├── requirements.txt               # Dependencies (torch, torch_geometric, langchain, etc.)
└── README.md                      # The most important file!
