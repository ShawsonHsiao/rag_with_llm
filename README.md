# Local RAG Knowledge Retrieval

This repository implements a **local Retrieval-Augmented Generation (RAG) pipeline** for document-based question answering.

The system retrieves relevant document chunks and uses a **local LLM** to generate answers based only on the retrieved context.

The project is designed to run fully locally without calling external APIs.

---

# Workflow Summary

1. **Clone the repository**
2. **Install dependencies** using `environment.txt`
3. **Ensure the default directory structure is preserved**
4. **Run the program (`main.py`)**

---

# Project Setup

Create an environment before installing dependencies.

Install the required Python dependencies using the provided environment file.

`pip install -r environment.txt`

# Project Directory Structure

Please do not modify the default directory layout.
Some scripts use relative paths, so changing folder names may cause errors.

```bash
RAG/
├── data/raw/ # Raw documents for indexing
├── models/ # Embedding model and LLM
├── src/ # Supporting source code
├── vec_database/ # Stored vector index and metadata
├── main.py # Program entry point
└── requirements.txt # Python dependencies
```

# Models

The project uses **Qwen3-Embedding-0.6B** as the embedding model and **Qwen3-4B-Instruct** as the language model (LLM).

If the models are not already present, they will be automatically downloaded when main.py is executed.

# Running the Project

```bash
cd RAG
python main.py
```
