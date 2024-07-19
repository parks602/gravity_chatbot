from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, tiktoken

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=tiktoken_len
    )

def load_documents_into_database(emmodel_name, documents_path):
    EMBEDDING = HuggingFaceEmbeddings(
        model_name= emmodel_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents) 
    
    if os.path.exists("/database/chroma.db"):
        db = Chroma(persist_directory="./database/chroma.db", embedding_function=EMBEDDING)
    else:
        db = Chroma.from_documents(documents, EMBEDDING, persist_directory = "./database/chroma.db")
    
    db.add_documents(documents)
    db.persist()
    return db

def load_documents(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs

def main(emmodel_name:str="BAAI/bge-m3", documents_path:str="./pdf_data"):
    load_documents_into_database(emmodel_name, documents_path)
    
if __name__ == "__main__":
    main()
    