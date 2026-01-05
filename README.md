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

The project targets **Python 3.13**, uses **Conda** for the base runtime, and **uv** for Python dependency locking via `pyproject.toml` and `uv.lock`.

---

## Quick Start (TL;DR)

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yaml
conda activate langchain

# 2. Install Python dependencies (tracked by uv)
uv sync

# 3. Register the Jupyter kernel (one-time)
python -m ipykernel install --user --name langchain --display-name "Python (langchain)"

# 4. Launch JupyterLab
jupyter lab
```

Open any notebook and select the **Python (langchain)** kernel.

---

## Repository Structure

```
.
├── environment.yaml
├── pyproject.toml
├── uv.lock
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
Modern tool-calling agents using `create_agent` and LangGraph-style state.

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

## Environment & Dependency Model

- **Conda** manages Python 3.13, Jupyter, and system libraries.
- **uv** manages Python package resolution and locking.

This separation keeps installs reproducible and avoids dependency drift.

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

- Always run notebooks **top to bottom** after a kernel restart.
- Invoke agents using the **messages** state format.
- Ensure you are using the **Python (langchain)** kernel.
- Do not install packages manually with `pip`; use `uv`.

---

## Requirements

- macOS or Linux
- Conda (Miniconda or Anaconda)
- Python 3.13
- OpenAI API key

---

## License

Educational / experimental use.