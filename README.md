# NEP-Insight-Bot
An AI-powered chatbot that provides precise, context-aware answers about India’s National Education Policy (NEP 2020) using semantic search and a large language model.

# 🤖 NEP Chatbot

A **smart AI assistant** for India’s **National Education Policy (NEP 2020)**. Ask questions about NEP, and get **precise, context-aware, and structured answers** instantly.

---

## 🔹 Features

- **PDF-based Knowledge:** Loads the full NEP 2020 document and splits it into semantic chunks.
- **Semantic Search:** Uses FAISS to retrieve the most relevant sections for your query.
- **Intelligent Answers:** Powered by **Gemma 2B**, a 2-billion parameter language model, with **retrieval-augmented generation**.
- **Example-Guided Responses:** Learns from curated Q&A examples for clear, structured, and bullet-point answers.
- **Modern Chat Interface:** Built with **Gradio** for an interactive, dark-mode chat experience.
- **Memory Efficient:** Uses **4-bit quantization** to run large language models on modest hardware.

---

## 🔹 How It Works

1. **Document Loading:** NEP 2020 PDF is loaded and split into overlapping chunks.
2. **Embedding & Indexing:** Text chunks are converted into embeddings using **Sentence Transformers** and indexed with **FAISS**.
3. **Query Processing:** User queries are matched to the most relevant document chunks.
4. **Answer Generation:** Gemma 2B generates responses using the retrieved context and example prompts.
5. **Interactive UI:** Users interact through a sleek **Gradio chat interface**.

---

## 🔹 Installation

```bash
pip install -U transformers langchain langchain-community accelerate bitsandbytes huggingface_hub gradio pypdf faiss-cpu sentence-transformers
````

Login to HuggingFace:

```python
from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")
```

---

## 🔹 Usage

```python
# Launch the Gradio chat UI
demo.launch()
```

Ask any question about NEP 2020, and the chatbot will provide a detailed answer with context, examples, and bullet points.

---

## 🔹 Example Questions

* What are the multiple entry and exit options in higher education?
* How does NEP promote multidisciplinary learning?
* What is the Academic Bank of Credit (ABC)?
* How does NEP support vocational training?

---

## 🔹 Tech Stack

* **Python**
* **LangChain** (orchestration)
* **FAISS** (vector search)
* **HuggingFace Transformers** (Gemma 2B model)
* **Sentence Transformers** (embeddings)
* **Gradio** (UI)

---

## 🔹 License

This project is released under the **MIT License**.

---

### 🌟 Bonus

This chatbot is a **Retrieval-Augmented Generation (RAG)** system:

* Combines **semantic search** and **LLM generation**.
* Provides precise answers based on NEP 2020 content.
* Example-guided to produce structured, professional responses.

---

> Made with ❤️ by n1kozera
