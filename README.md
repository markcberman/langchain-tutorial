# LangChain + OpenAI (GPT-5 Mini) Learning & Reference Project

This repository is a **hands-on learning and reference project** for modern **LangChain (v1) patterns** using the **OpenAI Responses API** and **GPT-5 mini**.

It progresses from foundational concepts (prompts, parsers, memory, chains) to **production-relevant architectures**, including:

- Retrieval-Augmented Generation (RAG)
- Vector stores (ChromaDB)
- Structured LLM evaluation (LLM-as-a-judge)
- Agents and tool use

The repo is organized as a **sequence of Jupyter notebooks**, each building on the previous ones.

---

## Project Goals

This repository is designed to help you:

- Learn **LangChain v1 APIs** in a structured way
- Understand **how RAG systems are actually built**
- See how **LLMs can evaluate other LLMs**
- Develop intuition for **production tradeoffs** (retrieval quality, grounding, evaluation)
- Maintain a working **reference implementation** you can adapt to real systems

This is **not** a library or framework — it is an **executable curriculum + sandbox**.

---

## Environment Setup (Python 3.13)

This project intentionally targets **Python 3.13**.

Because FAISS does not reliably support Python 3.13, the project uses **ChromaDB** as the vector store.

### 1. Create and activate the Conda environment

```bash
conda env create -f environment.yaml
conda activate langchain
```

The environment explicitly installs (via `pip` inside Conda):

- LangChain core + integrations
- ChromaDB
- Jupyter tooling
- Pandas for CSV-based RAG

See `environment.yaml` for the authoritative dependency list.

---

### 2. Verify the environment

Run the following in Python or a notebook:

```python
import chromadb
from langchain_community.vectorstores import Chroma

print("ChromaDB is installed and working")
```

---

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-key-here"
```

This project uses the **OpenAI Responses API** via `langchain-openai`.

---

## Repository Structure & Notebook Guide

Each notebook focuses on **one conceptual layer** of LangChain.

They can be run independently, but are best read **in order**.

---

### L1 – Models, Prompts, and Output Parsers  
`L1-Model_prompt_parser_UPDATED.ipynb`

**Covers**
- Chat models (`ChatOpenAI`)
- Prompt templates
- Output parsing
- Deterministic vs generative outputs

**Why it matters**  
LLMs are stochastic by default; correctness requires **explicit structure**.

---

### L2 – Memory  
`L2-Memory_UPDATED.ipynb`

**Covers**
- Conversation memory
- Stateful vs stateless chains
- Memory tradeoffs

**Why it matters**  
Memory affects correctness, cost, and hallucination risk.

---

### L3 – Chains  
`L3-Chains_UPDATED.ipynb`

**Covers**
- Runnable composition
- Prompt → model → parser pipelines
- Data flow through chains

**Why it matters**  
Chains are the backbone of production LangChain systems.

---

### L4 – Question Answering & Retrieval  
`L4-QnA_UPDATED.ipynb`

**Covers**
- Document loading
- Text splitting
- Vector stores
- Retrieval-based QA

**Why it matters**  
This notebook introduces **RAG** and shows how retrieval reframes the LLM’s role.

---

### L5 – Evaluation (LLM-as-a-Judge)  
`L5-Evaluation_UPDATED.ipynb`

**Covers**
- LLM-based evaluation
- Explicit rubrics
- Structured outputs with Pydantic

**Why it matters**  
Correctness in RAG means **support from retrieved context**, not plausibility.

---

### L5.5 – End-to-End RAG + Evaluation (CSV + Chroma)  
`L5_5_rag_and_rag_eval_outdoor_catalog.ipynb`

**Covers**
- CSV knowledge base ingestion (`OutdoorClothingCatalog_1000.csv`)
- ChromaDB indexing
- RAG answer generation with GPT-5 mini
- Groundedness evaluation using a separate judge chain

**Architecture**
```
CSV → Documents → Chroma
     → Retriever → Answer LLM
     → Judge LLM → Structured verdict
```

---

### L6 – Agents  
`L6-Agents_UPDATED.ipynb`

**Covers**
- Tool use
- Agent loops
- LLM-driven decision-making

**Why it matters**  
Agents introduce autonomy and risk and require stronger guardrails.

---

## Vector Store Choice: ChromaDB

ChromaDB is used because it:

- Works reliably on **Python 3.13**
- Supports local, persistent storage
- Integrates cleanly with LangChain
- Avoids FAISS binary compatibility issues

Chroma persists data locally (for example: `.chroma_outdoor_catalog/`).

If documents or embeddings change, rebuild the index:

```bash
rm -rf .chroma_outdoor_catalog
```

---

## Design Philosophy

This repo intentionally demonstrates best practices:

- Separate **answer** and **evaluation** prompts
- Explicit evaluation rubrics
- Deterministic generation for debugging
- Modular chains instead of monolithic prompts

Some notebooks trade conciseness for **clarity and inspectability** — deliberately.

---

## What This Repo Is Not

- ❌ A reusable Python package
- ❌ A production service
- ❌ A benchmark suite

It *is* a strong foundation for building those things.

---

## Summary

Working through these notebooks will give you a practical understanding of:

- LangChain v1 architecture
- Modern RAG systems
- LLM-based evaluation
- Retrieval quality vs answer quality
- Why architectural separation matters
