# FILE: indexer.py
# DESCRIPTION: Loads 5 AI papers (PDFs), splits them into chunks,
#              generates embeddings, and stores them in a local ChromaDB vector database.
# AUTHOR: Marta Benitez Aguilar
# DATE: 2026-06-23

import os                              #To build file paths that work on any OS
from dotenv import load_dotenv         #To load the API key from the .env file

from langchain_community.document_loaders import PyPDFLoader   #To read PDF files (simple text, no complex tables etc pdf plumber is other option outside langchain)
from langchain_text_splitters import RecursiveCharacterTextSplitter  #To split text into chunks (it already respects the phrase, paragraphs and words structure, not only characteres)
from langchain_huggingface import HuggingFaceEmbeddings   #To convert text to vectors (multiple options but HF free, OpenAI is another possibility but not free)
from langchain_community.vectorstores import Chroma                 #To store vectors in ChromaDB (ChromaDB is also going to search the similar vector, supabase with SQL is a virtual option, but it requires more setup for a little project)

#LOAD ENVIRONMENT VARIABLES -> not necessary for this code 
#load_dotenv()   #Reads the .env file and makes ANTHROPIC_API_KEY available in the environment

#CONFIGURATION
DATA_DIR = "data"           #Folder where the 5 PDFs live
CHROMA_DIR = "chroma_db"    #Folder where ChromaDB will store the vectors (created automatically)


def run_indexer(chunk_size, chunk_overlap): 
    #Each chunk will be max CHUNK_SIZE characters long (he is going to stop before but the nearest taking into account the paragraphs, words, etc) (1 token = aprox 4caract)
    #CHUNK_OVERLAP characters overlap between consecutive chunks 

    #STEP 1: LOAD ALL PDFs
    print("Loading PDFs...") #info step
    papers = []   #Empty list where we are going to store the text of all docs

    for filename in os.listdir(DATA_DIR):                        #Loop through every file in data/
        if filename.endswith(".pdf"):                            #Only process PDF files
            filepath = os.path.join(DATA_DIR, filename)          #Build the full path: data/filename.pdf
            loader = PyPDFLoader(filepath)                       #Create a loader for this PDF
            papers.extend(loader.load())                      #Add all pages of this PDF to the list

    print(f"Loaded {len(papers)} pages from {DATA_DIR}/")    #Show how many pages total

    #STEP 2: SPLIT INTO CHUNKS
    print("Splitting into chunks...") #info step
    splitter_def = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        
        chunk_overlap=chunk_overlap   
    )
    chunks = splitter_def.split_documents(papers)   #Split all pages into chunks
    print(f"Created {len(chunks)} chunks")         #Show how many chunks total

    #STEP 3: GENERATE EMBEDDINGS AND STORE IN CHROMADB

    #NOTA: you can chose other models (see list in MTEB leaderboard for example)
    #      but it is important to use the same model to translate the query 
    #      More parameters are not always = more precise, it captures more details but with a higher computational cost

    print("Generating embeddings and storing in ChromaDB...") #info step
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"   #Small, fast embedding model, runs locally, 384 values
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,              #The chunks to embed and store
        embedding=embedding_model,     #The model that converts text to vectors
        persist_directory=CHROMA_DIR   #Where to save the database on disk
    )

    print(f"Done. Vector database saved in '{CHROMA_DIR}/'")

# ENTRY POINT: allows running this script directly from the terminal
if __name__ == "__main__":
    run_indexer(chunk_size=500, chunk_overlap=50)