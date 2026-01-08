# LangChain Learning Notebooks

This repository contains a set of **hands-on Jupyter notebooks** demonstrating modern **LangChain v1.x** patterns using OpenAI models, including:

- Prompt parsing
- Memory
- Chains and runnables
- Question Answering (Q&A)
- Retrieval-Augmented Generation (RAG)
- Evaluation
- Tagging and extraction
- Tool calling and routing
- Conversational and tool-using agents

The project targets **Python 3.13**, uses **Conda** for the runtime (Python + Jupyter + system libs), and **uv** for Python dependency resolution + locking (`pyproject.toml` + `uv.lock`).

---

## Two audiences, two workflows

### If you're a user (you just want to run the notebooks)
You will:
1. Create the Conda env from `environment.yaml`
2. Install Python packages into that env using **the exported requirements lock** (`requirements.locked.txt`)
3. Run JupyterLab from that env

### If you're a developer (you will change dependencies)
You will:
1. Use `uv add ...` to change dependencies (updates `pyproject.toml` and `uv.lock`)
2. Export a pip-compatible lock (`requirements.locked.txt`)
3. Sync that lock into the Conda runtime env

Why the extra export step? `uv.lock` is a uv project lockfile (TOML). `uv pip sync` expects a **requirements.txt-style** lock, not `uv.lock`.

---

## Quick Start (users)

```bash
# 1) Create + activate the conda runtime env
conda env create -f environment.yaml
conda activate langchain

# 2) Install the pinned Python dependencies into THIS conda env
# (requirements.locked.txt is generated from uv.lock and committed)
uv pip sync requirements.locked.txt

# 3) Register the Jupyter kernel (one-time)
python -m ipykernel install --user --name langchain --display-name "Python (langchain)"

# 4) Launch JupyterLab
python -m jupyter lab
```

Open any notebook and select the **Python (langchain)** kernel.

> If `python -m jupyter lab` fails with “No module named jupyter”, install JupyterLab into the conda env:
> `conda install -c conda-forge jupyterlab ipykernel`

---

## Developer workflow (changing dependencies)

### 0) Prereqs
- Have the conda env active when working on deps (so uv can pin to the same Python):
  ```bash
  conda activate langchain
  ```

### 1) Add / update a dependency
```bash
uv add langchain-tavily
uv lock
```

### 2) Export a pip-compatible lockfile for the conda runtime
```bash
uv export --format requirements.txt --output-file requirements.locked.txt
```

### 3) Apply the lockfile to the active conda env
```bash
conda activate langchain
uv pip sync requirements.locked.txt
```

### 4) Commit the right files
Commit these when dependencies change:
- `pyproject.toml`
- `uv.lock`
- `requirements.locked.txt`

Commit/update this when **conda-level** dependencies change (Python version, JupyterLab, CUDA/PyTorch, etc.):
- `environment.yaml` (recommended export: `conda env export --from-history > environment.yaml`)

---

## Repository Structure

```
.
├── environment.yaml
├── pyproject.toml
├── uv.lock
├── requirements.locked.txt
├── README.md
├── main.py
├── Data.csv
├── OutdoorClothingCatalog_1000.csv
├── L1-Model_prompt_parser_UPDATED.ipynb
├── L2-Memory_UPDATED.ipynb
├── L3-Chains_UPDATED.ipynb
├── L4-QnA_UPDATED.ipynb
├── L5-Evaluation_UPDATED.ipynb
├── L5_5_rag_and_rag_eval_outdoor_catalog.ipynb
├── L6-Agents_UPDATED.ipynb
├── L7-openai_functions_student.ipynb
├── L8-lcel-student.ipynb
├── L9-function-calling-student.ipynb
├── L9_5_langchain_tool_calling_options.ipynb
├── L10-tagging-and-extraction-student.ipynb
├── L10_5_tagging_extraction_with_structured_output.ipynb
├── L11-tools-routing-apis-student.ipynb
└── L12-conversational-agents.ipynb
```

---

## Notebook Overview

### L1 – Model & Prompt Parsing
Introduces prompt templates, output parsers, and structured prompting patterns.

### L2 – Memory
Demonstrates conversational memory patterns and state handling in LangChain v1.

### L3 – Chains
Shows how to build and compose chains using LangChain’s runnable interface.

### L4 – Q&A
Implements question-answering flows with prompts and retrievers.

### L5 – Evaluation
Covers response evaluation using LangChain evaluators.

### L5.5 – RAG + Evaluation
An end-to-end Retrieval-Augmented Generation pipeline with evaluation on an outdoor catalog dataset.

### L6 – Agents
Modern tool-calling agents using LangGraph-style state.

### L7 – OpenAI Functions (Legacy → Modern Context)
Explains legacy OpenAI function-calling concepts for historical context and contrasts them with modern tool calling.

### L8 – LCEL (LangChain Expression Language)
Introduces LCEL syntax for composing prompts, models, and parsers declaratively.

### L9 – Function Calling
Explores structured function / tool calling patterns from a learning perspective.

### L9.5 – Tool Calling Options
Deep dive into different tool-binding and invocation strategies in LangChain v1.x.

### L10 – Tagging & Extraction
Demonstrates entity tagging and information extraction from text.

### L10.5 – Structured Output
Shows how to enforce schemas and structured outputs using Pydantic models.

### L11 – Tool Routing & APIs
Routes user requests to different tools and APIs dynamically.

### L12 – Conversational Agents
Builds multi-turn conversational agents with tool use and memory.

---

## API Keys

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

Or via a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

---

## Troubleshooting

### “jupyter: command not found”
Install Jupyter into the conda env:
```bash
conda install -c conda-forge jupyterlab ipykernel
```
And launch via:
```bash
python -m jupyter lab
```

### Notebook can’t import a package you just installed
You are almost certainly on the wrong kernel. In Jupyter: **Kernel → Change Kernel → Python (langchain)**.

To confirm inside a notebook cell:
```python
import sys
sys.executable
```

### Don’t use this (common mistake)
```bash
uv pip sync uv.lock
```
`uv.lock` is not a requirements file. Use:
- `uv sync` (for uv-managed project environments), or
- `uv export ...` + `uv pip sync requirements.locked.txt` (for conda runtime envs)

---

## Requirements

- macOS, Linux, or WSL2
- Conda (Miniconda or Anaconda)
- uv installed
- OpenAI API key
