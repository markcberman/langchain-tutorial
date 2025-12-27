# LangChain + OpenAI (GPT-5 Mini) Learning Project

This project demonstrates **Retrieval-Augmented Generation (RAG)** and **LLM-based evaluation**
using **LangChain**, **OpenAI (GPT-5 mini)**, and **ChromaDB** as the local vector store.

> **Important**: We intentionally use **ChromaDB** instead of FAISS because FAISS does not
reliably support Python 3.13. Chroma works correctly on Python 3.13 when installed via `pip`.

---

## Environment Setup (Python 3.13)

### 1. Create and activate the Conda environment

```bash
conda env create -f environment_chroma_py313.yaml
conda activate langchain
```

This environment:
- Uses **Python 3.13**
- Installs **ChromaDB via pip**
- Includes LangChain core + community integrations
- Is ready for Jupyter notebooks

---

### 2. Verify ChromaDB installation

Run the following in a Python shell or notebook:

```python
import chromadb
from langchain_community.vectorstores import Chroma

print("ChromaDB is installed and working")
```

If this succeeds, your environment is correctly set up.

---

## Why ChromaDB?

We use **ChromaDB** as the vector store because:

- ✅ Works reliably with **Python 3.13**
- ✅ Supports **local, persistent** vector storage
- ✅ Integrates cleanly with LangChain
- ❌ Avoids native binary issues seen with FAISS on newer Python versions

Chroma stores its data locally (for example: `.chroma_outdoor_catalog/`).
If you change documents or embeddings, delete this directory to rebuild the index.

---

## Running the Notebooks

```bash
jupyter lab
```

Make sure the active kernel is **Python (langchain)**.

The notebooks demonstrate:

- Loading a CSV knowledge base into LangChain `Document`s
- Indexing documents with **Chroma**
- RAG answer generation using **GPT-5 mini**
- Groundedness evaluation using an **LLM-as-a-judge** with a strict rubric

---

## Architecture Overview

```
CSV → LangChain Documents → Chroma Vector Store
     → Retriever → GPT-5 mini (answer generation)
     → GPT-5 mini (groundedness evaluation)
```

---

## Notes

- Uses **OpenAI Responses API**
- Uses **LangChain v1 APIs**
- Default deterministic model: **gpt-5-mini**
- Requires `OPENAI_API_KEY` for live LLM calls

---

## Troubleshooting

### Chroma feels “stale” after changes
Delete the persistence directory and re-run the notebook:

```bash
rm -rf .chroma_outdoor_catalog
```

### Import errors
Ensure:
- You activated the `langchain` Conda environment
- Jupyter is using the correct kernel

---
