# CGLM-MR: Cognitive Graph-based Long-term Memory and Meta-reasoning Architecture 🧠

An advanced cognitive memory system for LLM agents that dynamically organizes, evolves, and reasons over long-term memories using causal graphs and meta-reasoning.

> **Note:** This repository contains the implementation of CGLM-MR, a significant evolution of the A-Mem framework. It introduces causal logic topology and biological-inspired dynamic pruning to solve the "memory bloat" and logical hallucination issues in long-term agent interactions.

## Introduction 🌟

While Large Language Model (LLM) agents excel in complex open-ended tasks, traditional Retrieval-Augmented Generation (RAG) and simple context windows fail to maintain logical consistency over long-term interactions. Even previous agentic memory systems like A-Mem, which use semantic links based on Zettelkasten principles, struggle with "causal entanglement" and "memory bloat" over time.

CGLM-MR elevates agent memory from a flat semantic note network to a structured cognitive graph with causal depth and biological-inspired dynamic pruning. It simulates human brain functions like synaptic pruning to maintain cognitive efficiency and uses Dual Process Theory (System 1 & System 2) to balance fast intuitive retrieval with slow, deliberate causal reasoning.

## Key Features ✨

- 🧠 **Disentangled Cognitive Graph**: Constructs a heterogeneous dynamic graph with Semantic, Temporal, Causal, and Entity edges.
- 🔗 **Automatic Causal Extraction & Verification**: Upgrades simple semantic associations to causal logic topologies using counterfactual intervention verification.
- ✂️ **Explainable Graph Evolution (Synaptic Pruning)**: Utilizes dynamic Graph Neural Networks (GNN) to simulate synaptic pruning, continuously optimizing the graph topology and filtering noise.
- ⚖️ **Meta-Reasoning Controller**: Implements Dual Process Theory to dynamically switch between System 1 (fast semantic retrieval) and System 2 (deep causal graph traversal) for optimal efficiency and accuracy.
- 📉 **High Efficiency**: Reduces average token consumption by over 15% while significantly improving multi-hop reasoning capabilities.

## Framework 🏗️

The architecture integrates LLM capabilities with Structured Causal Models (SCM) and Dynamic GNN optimization:
1. **Memory Node Construction**: Deconstructs interactions into atomic facts and OTAR (Observation, Thought, Action, Result) tuples.
2. **Causal Hypothesis & Verification**: Generates causal links verified through counterfactual tests.
3. **Graph Evolution**: Periodically prunes redundant or weak connections using stability-plasticity balance.
4. **Meta-Reasoning Execution**: Autonomously determines cognitive load based on task complexity and logical consistency.

## How It Works 🛠️

When a new memory is added to the system:
1. Parses the interaction into atomic facts, entities, and an OTAR tuple.
2. Establishes temporal, semantic, and entity links with historical memory nodes.
3. Automatically extracts and verifies causal relationships using LLM-driven counterfactual intervention.
4. Periodically evolves the graph using a GNN-based synaptic pruning algorithm to remove redundant links and consolidate knowledge.
5. Retrieves information using a Meta-Reasoning controller that balances intuitive search (System 1) with deep logical inference (System 2).

## Results 📊

Empirical experiments conducted on the LoCoMo long-term conversational dataset across six foundation models demonstrate that CGLM-MR significantly outperforms the A-Mem baseline. It shows remarkable improvements in **Multi-hop Reasoning** and **Temporal Awareness** tasks, proving its capability in maintaining long-term survival and meta-cognitive abilities for general agents.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/liuzongquan/CGLM-MR.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv cglm-env
source cglm-env/bin/activate  # Linux/Mac
cglm-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n cglm-env python=3.9
conda activate cglm-env

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments on the LoCoMo dataset:
```bash
python test_cglm_mr.py 
```

**Categories Information:** The LoCoMo dataset contains the following categories:
* Category 1: Multi-hop
* Category 2: Temporal
* Category 3: Open-domain
* Category 4: Single-hop
* Category 5: Adversarial

For more details about the categories, please refer to [this GitHub issue](https://github.com/snap-research/locomo/issues/6). 

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
