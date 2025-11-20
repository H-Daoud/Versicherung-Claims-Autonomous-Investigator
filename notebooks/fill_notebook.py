import json
import os

# Content of the Notebook cells
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🕵️‍♂️ Autonomous Claims Investigator: Prototype Demo\n",
    "\n",
    "This notebook demonstrates the **'Detective Agent'** (Graph Neural Network logic) and the **'Compound System'** workflow.\n",
    "\n",
    "### Scenario\n",
    "We are analyzing a cluster of insurance claims. Most look normal, but a hidden **'Fraud Ring'** exists where multiple claimants share the same suspicious Repair Shop and Doctor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Setup & Imports\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import random\n",
    "\n",
    "# Set visualization style\n",
    "%matplotlib inline\n",
    "plt.rcParams['figure.figsize'] = (12, 8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Generate Synthetic Knowledge Graph\n",
    "We simulate a database of Claims, Policy Holders, and Service Providers (Shops/Doctors)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_fraud_ring_graph():\n",
    "    G = nx.Graph()\n",
    "    \n",
    "    # --- Legitimate Clusters (Normal claims) ---\n",
    "    # People just connected to their own claims and random shops\n",
    "    for i in range(1, 6):\n",
    "        person = f\"Person_{i}\"\n",
    "        claim = f\"Claim_{i}\"\n",
    "        shop = f\"Shop_{i}\"\n",
    "        \n",
    "        G.add_node(person, type='Person', color='blue')\n",
    "        G.add_node(claim, type='Claim', color='green')\n",
    "        G.add_node(shop, type='Shop', color='gray')\n",
    "        \n",
    "        G.add_edge(person, claim, relation='FILED')\n",
    "        G.add_edge(claim, shop, relation='REPAIRED_AT')\n",
    "\n",
    "    # --- The FRAUD RING (Organized Crime) ---\n",
    "    # Multiple people connected to ONE shady shop and ONE shady doctor\n",
    "    shady_shop = \"⚠️ BAD_SHOP_99\"\n",
    "    shady_doc = \"⚠️ BAD_DOC_99\"\n",
    "    \n",
    "    G.add_node(shady_shop, type='Shop', color='red')\n",
    "    G.add_node(shady_doc, type='Doctor', color='red')\n",
    "    \n",
    "    fraudsters = ['Fraudster_A', 'Fraudster_B', 'Fraudster_C']\n",
    "    \n",
    "    for f in fraudsters:\n",
    "        claim_id = f\"Claim_{f}\"\n",
    "        \n",
    "        G.add_node(f, type='Person', color='orange')\n",
    "        G.add_node(claim_id, type='Claim', color='orange')\n",
    "        \n",
    "        # They file a claim\n",
    "        G.add_edge(f, claim_id, relation='FILED')\n",
    "        \n",
    "        # BUT they all go to the SAME shady providers\n",
    "        G.add_edge(claim_id, shady_shop, relation='REPAIRED_AT')\n",
    "        G.add_edge(f, shady_doc, relation='TREATED_BY')\n",
    "        \n",
    "    return G\n",
    "\n",
    "G = generate_fraud_ring_graph()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. The \"Detective Agent\" View (GNN Logic)\n",
    "A standard SQL query might miss this because the claims look separate. \n",
    "**Graph Analysis** reveals the highly connected \"hub\" nodes (The Bad Shop and Bad Doctor) shown in **RED**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualization Logic\n",
    "def plot_graph(G):\n",
    "    pos = nx.spring_layout(G, k=0.3, iterations=50)\n",
    "    \n",
    "    # Extract colors from node attributes\n",
    "    node_colors = [G.nodes[n]['color'] for n in G.nodes]\n",
    "    \n",
    "    plt.figure(figsize=(10, 8))\n",
    "    nx.draw(\n",
    "        G, pos, \n",
    "        with_labels=True, \n",
    "        node_color=node_colors, \n",
    "        node_size=1500, \n",
    "        font_size=9, \n",
    "        font_weight='bold',\n",
    "        edge_color='gray',\n",
    "        alpha=0.8\n",
    "    )\n",
    "    \n",
    "    # Create a legend\n",
    "    legend_labels = {'Normal Claim': 'green', 'Fraud Ring': 'red', 'Suspect': 'orange', 'Person': 'blue'}\n",
    "    for label, color in legend_labels.items():\n",
    "        plt.scatter([], [], c=color, label=label, s=100)\n",
    "    \n",
    "    plt.legend(loc='upper left')\n",
    "    plt.title(\"Graph Analysis: Detecting the Fraud Ring (Red Hubs)\", fontsize=15)\n",
    "    plt.show()\n",
    "\n",
    "plot_graph(G)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. The \"Lawyer Agent\" (RAG Policy Check)\n",
    "Now that the fraud ring is detected, we check the policy coverage for one of the fraudulent claims."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "claim_text = \"Claim_Fraudster_A: Front bumper damage reported at 2 AM. Towing service used: QuickFix Motors.\"\n",
    "\n",
    "# Simulated RAG Output\n",
    "policy_context = \"\"\"\n",
    "POLICY EXCLUSION 4.1: The insurer shall not be liable for loss or damage if the vehicle \n",
    "is repaired at an unauthorized service provider listed in the 'Blacklist Registry'.\n",
    "\"\"\"\n",
    "\n",
    "print(f\"--- Analyzing Policy Coverage for: {claim_text} ---\")\n",
    "print(f\"🔍 RETRIEVED CONTEXT: {policy_context}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. The \"Judge Agent\" (Final Verdict)\n",
    "The system combines the Graph Evidence (Fraud Ring) + Policy Evidence (Exclusion) to make a decision."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_verdict = \"\"\"\n",
    "⚖️ FINAL DECISION: REJECTED\n",
    "\n",
    "REASONING:\n",
    "1. FRAUD DETECTED: Claimant is part of a known fraud ring connected to 'BAD_SHOP_99'. (Confidence: 98%)\n",
    "2. POLICY EXCLUSION: The repair shop is on the unauthorized provider list (Clause 4.1).\n",
    "\n",
    "ACTION: Flag for Special Investigation Unit (SIU).\n",
    "\"\"\"\n",
    "\n",
    "print(final_verdict)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Write the file
output_path = os.path.join("notebooks", "prototype_demo.ipynb")
with open(output_path, "w") as f:
    json.dump(notebook_content, f, indent=1)

print(f"✅ Successfully filled {output_path}")
