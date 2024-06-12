import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
import pathlib
import logging
import streamlit as st

# Set logging
logging.basicConfig(level=logging.INFO)

# Initialize tokenizer and models
ACCESS_TOKEN = ""  # Ensure you have your access token

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

qa_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, token=ACCESS_TOKEN)
qa_model.eval()

# Load FAISS index
index_path = pathlib.Path("/content/drive/MyDrive/fyp/WikiBuddy-main/saved-index-faiss")  # Update the path as necessary
embeddings_db = FAISS.load_local(
    index_path, embedding_model, allow_dangerous_deserialization=True
)
retriever = embeddings_db.as_retriever(search_kwargs={"k": 3})

# Setup memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def generate_answer_faiss(question: str):
    results = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in results])

    prompt = f"Using the information contained in the context, give a detailed answer to the question. Context: {context}. Question: {question}"
    
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer.encode(
        formatted_prompt, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = qa_model.generate(
            input_ids=inputs,
            max_new_tokens=250,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response[len(formatted_prompt):]  # remove input prompt from response
    response = response.replace("<eos>", "")  # remove eos token
    return response

def generate_answer_pdf(question: str, context: str):
    if context == None or context == "":
        prompt = f"Give a detailed answer to the following question. Question: {question}"
    else:
        prompt = f"Using the information contained in the context, give a detailed answer to the question. Context: {context}. Question: {question}"
    
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer.encode(
        formatted_prompt, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = qa_model.generate(
            input_ids=inputs,
            max_new_tokens=250,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response[len(formatted_prompt):]  # remove input prompt from response
    response = response.replace("<eos>", "")  # remove eos token
    return response

def ask(message):
    return generate_answer_faiss(message)

# Document Loading and Splitting Functions
def load_and_split_pdf(file_path):
    loaders = [PyPDFLoader(file_path)]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=12)
    docs = text_splitter.split_documents(pages)
    return docs

def create_faiss_index(docs):
    faiss_db = FAISS.from_documents(docs, embedding_model, distance_strategy=DistanceStrategy.DOT_PRODUCT)
    return faiss_db

# Streamlit app with two different pages
def main():
    st.sidebar.title("WikiBuddy")
    page = st.sidebar.selectbox("Choose a page", ["Chatbot", "PDF QA"])

    if page == "Chatbot":
        st.title("Chatbot with Gemma Model")
        st.write("Ask me anything!")
        question = st.text_input("Your question:")
        if question:
            response = ask(question)
            st.write(response)

    elif page == "PDF QA":
        st.title("Upload PDF and Ask Questions")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            docs = load_and_split_pdf("uploaded_file.pdf")
            faiss_db = create_faiss_index(docs)

            question = st.text_input("Your question:")
            if question:
                retrieved_docs = faiss_db.similarity_search(question, k=5)
                context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
                response = generate_answer_pdf(question, context)
                st.write(response)

                st.write("For this answer I used the following documents:")
                for doc in retrieved_docs:
                    st.write(doc.metadata)

if __name__ == "__main__":
    main()
