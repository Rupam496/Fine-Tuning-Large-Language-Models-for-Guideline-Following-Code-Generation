# Guideline-Aware Code Generation

This project focuses on improving **LLM-based code generation** so that generated code follows predefined coding guidelines.

The pipeline combines **guideline retrieval, LLM-based evaluation, and fine-tuning/GRPO** to improve guideline adherence.

## Pipeline

```text
Coding Guidelines
       ↓
Guideline Chunking
       ↓
Embedding & Vector Store
       ↓
Relevant Guideline Retrieval
       ↓
Code Generation
       ↓
Guideline-Based Evaluation
       ↓
Fine-Tuning / GRPO
       ↓
Improved Code Generation
```

## Key Components

* **Guideline Retrieval** – Retrieves relevant coding guidelines using embeddings.
* **Code Generation** – Generates code using the base and fine-tuned models.
* **LLM Judge** – Evaluates generated code based on the guidelines.
* **GRPO Training** – Uses guideline-based rewards to improve the model.
* **Output Comparison** – Compares base-model and fine-tuned-model outputs.


## Technologies

* Python
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Embedding-based Retrieval
* Fine-Tuning
* GRPO
* LLM-based Evaluation
* Git & GitHub

## Outputs

The repository contains generated outputs from both models:

```text
base_model_output/
finetuned_model_output/
```

These outputs can be used to compare **guideline adherence between the base and fine-tuned models**.
