# FinChat

Developing an AI-powered personal financial advisor by fine-tuning base models and integrating RAG-based systems for optimized, real-time financial advice.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [How to Build the Code](#how-to-build-the-code)
5. [How to Run](#how-to-run)
6. [Examine the Results](#examine-the-results)
7. [Expected Output](#expected-output)
8. [Extending the Code](#extending-the-code)
9. [Dataset Format](#dataset-format)
10. [Contributors](#contributors)

---

## Introduction

FinChat is an open-source financial chatbot combining fine-tuned Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). This project aims to democratize access to personalized financial advisory tools by integrating static, domain-specific expertise with real-time data retrieval.

## Features

- **Fine-Tuned LLMs**: Tailored to address personal finance-related queries.
- **RAG Integration**: Real-time context-based responses using vector databases.
- **Cost-Effective**: Open-source solution accessible to small businesses and individuals.
- **Dynamic Advisory**: Offers financial advice aligned with market changes.

---

## Installation

### Prerequisites
- Python 3.8+
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- OpenAI API access for fine-tuning
- LangChain library for RAG integration

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/pawanaditya85/FinChat.git
   cd FinChat
   ```
2. Create a Conda environment:
   ```bash
   conda create -n finchat_env python=3.8
   conda activate finchat_env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have the following key libraries:
   - `langchain`
   - `openai`
   - `faiss`
   - `numpy`
   - `pandas`

---

## How to Build the Code

### Fine-Tuning the Model
1. Prepare the dataset in JSONL format:
   - Structure:
     ```json
     {
       "prompt": "User query",
       "completion": "Expected AI response"
     }
     ```
2. Split the dataset into an 80:20 ratio for training and validation.
3. Fine-tune the base model (e.g., GPT-3.5) using the OpenAI API:
   ```bash
   openai api fine_tunes.create -t training_data.jsonl -v validation_data.jsonl
   ```
   For more information, refer to [OpenAI Fine-Tuning Docs](https://platform.openai.com/docs/guides/fine-tuning).

### Setting Up RAG
1. Embed documents using NVIDIA embedding models:
   ```python
   from langchain.embeddings import HuggingFaceEmbeddings
   ```
2. Create a FAISS vector database:
   ```python
   from langchain.vectorstores import FAISS
   vectorstore = FAISS.load_local("vector_db")
   ```
3. Integrate the LangChain framework for dynamic query processing.

---

## How to Run

### Running the LLM Fine-Tuned Model
1. Execute `FinChat_Finetuned.ipynb`:
   - Modify `openai_api_key` with your OpenAI credentials.
   - Input queries in the specified format.

### Running the RAG Model
1. Execute `FinChat_RAG.ipynb`:
   - Ensure the FAISS vector database is loaded.
   - Provide a query to retrieve real-time financial context.

### Running with Custom Datasets
1. Update the dataset folder (`/data`) with your structured JSONL files.
2. Modify the notebook's dataset path accordingly.

---

## Examine the Results

- Evaluation Metrics:
  - BLEU, ROUGE (Fine-Tuned LLM)
  - Faithfulness, Context Precision, Recall, and Relevancy (RAG Model)
- Use the provided evaluation scripts in the `Evaluation` folder.

---

## Expected Output

### Fine-Tuned Model
- Structured responses tailored to user queries.
- Average response time: ~2 seconds per query.

### RAG Model
- Dynamic responses with integrated real-time context.
- Average retrieval and generation time: ~3-5 seconds per query.

---

## Extending the Code

1. **New Datasets**:
   - Format the dataset as JSONL with fields: `prompt`, `completion`.
   - Add the new dataset to the `/data` folder.

2. **New Models**:
   - Replace the base LLM in `FinChat_RAG.ipynb` with your desired model (e.g., GPT-4, LLaMA).
   - Update embedding techniques for vector database generation.

3. **New Parameters**:
   - Update LangChain's prompt templates for modified query structures.

---

## Dataset Format

- **Personal Finance QA Dataset**: 
  ```json
  {
    "Question": "How do I manage my budget?",
    "Answer": "Create a monthly spending plan and track expenses."
  }
  ```
- **RAG Integration**: Documents are split into chunks and embedded using FAISS.

---

## Contributors

- **Pawan Aditya Man** (Leader)
- **Amarender Reddy Jakka**
- **Shreekar Kolanu**
