# LangChain Project Setup Guide

Follow these steps to set up and run the project:

## 1. Create and Activate a Virtual Environment

Open your terminal and run:

```bash
python -m venv chatbot_venv
chatbot_venv\Scripts\activate # For windows
source chatbot_venv\Scripts\activate # For Linux
```

## 2. Install Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## 3. Prepare Training Data

- Place your training data files inside the `data` folder.

## 4. (Optional) Change Data Path

If you want to use a different folder for your training data, you can change the `DATA_PATH` variable in the `config.py` file:

```python
# In config.py
DATA_PATH = "./data/{file_name}.parquet"  # Change this to your desired data folder path
```

## 5. Download and Install Ollama

Download Ollama from [https://ollama.com/download](https://ollama.com/download) and install it following the instructions for Windows.

## 6. Pull the Llama 3.1 8B Model

After installing Ollama, open a new terminal and run:

```bash
ollama pull llama3.1:8b
```

## 7. Start Ollama Server

Start the Ollama server by running:

```bash
ollama serve
```

Keep this terminal open while running your code.

## 8. Process and Upload Data

Run the data processing script:

```bash
python process_and_upload.py
```

## 9. Start the Chatbot

Finally, run the chatbot:

```bash
python chatbot.py
```

---

## Why These Technologies?

- **Ollama**: Provides a simple way to run large language models (LLMs) locally on your machine, making it easy to integrate advanced AI capabilities without relying on cloud APIs.
- **Llama 3.1 8B Model**: This open-source LLM offers strong performance for chat and question-answering tasks while being efficient enough to run on local hardware.
- **BAAI Embedder Model**: Used for generating high-quality vector embeddings from your text data, which improves the accuracy and relevance of information retrieval.
- **Chroma VectorDB**: A fast, open-source vector database that stores and searches embeddings efficiently, enabling quick retrieval of relevant documents for your chatbot.

These components together allow you to build a powerful, private, and efficient retrieval-augmented generation (RAG) chatbot system.

---

Your environment is now ready, and you can interact with