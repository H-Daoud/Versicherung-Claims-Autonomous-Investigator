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
