import os
import streamlit as st
import pickle
import time
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import Pipeline, AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint


#tokenizer = AutoTokenizer.from_pretrained("gpt3", clean_up_tokenization_spaces=True)

st.title("News Research Tool ")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
file_path="notebook/vector_index.pkl"

#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--devliakansaha4
#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" 
#"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"
main_placeholder=st.empty()
hf_api_token = "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"
# Replace with your valid API token

# Initialize the LLM with the correct parameters
#model = "gpt2-xl"  # Replace with your chosen model
#tokenizer = AutoTokenizer.from_pretrained("gpt3", clean_up_tokenization_spaces=True)
model_id="gpt2-xl"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
    huggingfacehub_api_token=hf_api_token
)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading.....STARTED.....")
    data = loader.load()

    # SPLIT DATA
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("TEXT SPLITTER.....STARTED.....")
    docs = text_splitter.split_documents(data)

    # Create embedding sentence-transformers/all-MiniLM-L6-v2
    model_name = "all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create a FAISS vector index using Hugging Face embeddings
    vectorindex_hf = FAISS.from_documents(docs, hf_embeddings)

    main_placeholder.text("Embedding Vector Started Building.....")
    time.sleep(2)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_hf, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)

        # **Fix 1: Use `invoke` instead of `__call__`**
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)  # Use invoke

        st.header("Answer")
        st.subheader(result["answer"])

        sources=result.get("Sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list=sources.split("\n")
            for source in sources_list:
                st.write(source)

        # **Fix 2 (Potential):**
        # Check