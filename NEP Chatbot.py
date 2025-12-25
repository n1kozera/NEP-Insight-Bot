# Install required packages
!pip install -U transformers langchain langchain-community accelerate bitsandbytes huggingface_hub gradio pypdf faiss-cpu sentence-transformers

# Login to HuggingFace
from huggingface_hub import login
login(token="")  # Replace with your actual token

# Load and split the NEP PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("Path")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Embed and index documents using FAISS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Load Gemma 2B model in 4-bit
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Semantic Retriever + Answering
from langchain.schema import Document

def fetch_relevant_context(query, k=7):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# Install required packages
!pip install -U transformers langchain langchain-community accelerate bitsandbytes huggingface_hub gradio pypdf faiss-cpu sentence-transformers

# Login to HuggingFace
from huggingface_hub import login
login(token="")  # Replace with your actual token

# Load and split the NEP PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("Path")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Embed and index documents using FAISS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Load Gemma 2B model in 4-bit
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Semantic Retriever + Answering
from langchain.schema import Document

def fetch_relevant_context(query, k=7):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def ask_nep_bot(query):
    context = fetch_relevant_context(query)

    examples = """
Example 1:
Question: What does the policy say about multiple entry and exit in higher education?
Answer: NEP 2020 introduces a modular degree system with multiple exit options: Certificate after 1 year, Diploma after 2, Bachelor’s after 3, and Bachelor’s with Research after 4 years. The Academic Bank of Credit (ABC) allows students to store credits and rejoin later, enabling true multiple entry and exit.

Example 2:
Question: How does NEP encourage multidisciplinary learning?
Answer: NEP 2020 pushes all higher education institutions to become multidisciplinary by 2040. Students will be allowed to take courses across arts, sciences, and vocational streams, eliminating rigid boundaries.
"""

    prompt = f"""
You are an intelligent assistant trained on India's National Education Policy (NEP 2020).
Answer the user's question clearly, precisely, and with deep understanding, even if the question is indirect or the keywords do not match exactly.
Try using bullet points and give specific explanation based on NEP 2020

Use the context below and take inspiration from the examples provided.

{examples}

Context:
{context}

Question: {query}
Answer:"""

    result = pipe(prompt, max_new_tokens=600, temperature=0.3, do_sample=False)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()

# Gradio Chat UI
import gradio as gr

def chat_interface(user_input, history):
    response = ask_nep_bot(user_input)
    history = history or []
    history.append((user_input, response))
    return "", history

custom_css = """
body, .gradio-container, .block, .chatbot {
    background-color: #121212 !important;
    color: #e0e0e0 !important;
}
.gr-chat-message {
    background-color: #1e1e1e !important;
    border-radius: 12px;
    padding: 12px;
    max-height: 800px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
    font-size: 16px !important;
}
textarea, input {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}
.chatbot {
    height: 1000px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("<h2 style='color:#00FFFF;'>🤖 NEP Chatbot</h2>")
    chatbot = gr.Chatbot(
        value=[("", "Hi! Ask me anything about the NEP 2020.")],
        bubble_full_width=False
    )
    msg = gr.Textbox(placeholder="Type your question here...", label="Your Question")
    state = gr.State([])

    msg.submit(chat_interface, [msg, state], [msg, chatbot])
    msg.submit(lambda: "", None, msg)

demo.launch()
