# FILE: query.py
# DESCRIPTION: Load the user query, generates embeddings, and search the similar vectors stored in the database ChromaDB.
#              Return the chunks whose vectors are similar (the paragraph and the page of the chunk is stored with the vector that contains the semantic information).
#              Calls the API with the query and the text that contains the relevant information to build the answer.
#              Print the answer of the LLM. 
# AUTHOR: Marta Benitez Aguilar
# DATE: 2026-06-23

#IMPORTS
import os 
from dotenv import load_dotenv
import anthropic
import subprocess

from langchain_huggingface import HuggingFaceEmbeddings   #To convert text to vectors (multiple options but HF free, OpenAI is another possibility but not free)
from langchain_community.vectorstores import Chroma                 #To store vectors in ChromaDB (ChromaDB is also going to search the similar vector, supabase with SQL is a virtual option, but it requires more setup for a little project)

#LOAD ENVIRONMENT VARIABLES 
load_dotenv()   #Reads the .env file and makes ANTHROPIC_API_KEY available in the environment

#CONFIGURATION
CHROMA_DIR = "chroma_db"    #Folder where ChromaDB have stored the vectors
TOP_K = 6             #N of chunks that will be given to the LLM to have the context (constant name = convention)

# #CHECK THE DATABASE: **only useful if you are using this script without RAG_evaluation as main**
# if not os.path.exists(CHROMA_DIR):
#     # run indexer first if the papers are not stored yet 
#     print("ChromaDB not found. Running indexer first...")
#     subprocess.run(["python", "src/indexer.py"])
# else:
#     print("You have the 5 most important papers about LLMs, RAG and AI available to answer your questions")

def run_query(query, top_k):

    #CHARGE THE DATABASE
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"   #SAME AS THE INDEXER
    )
    database = Chroma(
        persist_directory=CHROMA_DIR, 
        embedding_function=embedding_model
    )

    #SERACH THE IMPORTANT CHUNKS TO ANSWER THE QUERY
    context = database.similarity_search(query, k=top_k) #search the similar vectors that could be useful to answerthe query, returns a list of objects with TOP_K elements, storing the page_content and the metadata (name and page)

    #BUILD THE PROMPT TO THE LLM

    #1.Convert the context and the source into text 
    context_text = []
    for doc in context:
        context_text.append(f"{doc.page_content} Source: {doc.metadata['source']} In page: {doc.metadata['page']}") #you can only append one string

    context_text = "\n\n".join(context_text) #joins everything in the list between two spaces \n\n

    #2.Build the prompt with the query and the context 
    prompt = f"Answer the following question: {query} only with the following information {context_text}. \
            Give at the end the source and the page where you find the information to answer. \
            If the answer requires other information outside the context given answer: Sorry, but your question is out of the scope"

    #CALL THE API

    #Read the API key from the environment
    api_key = os.getenv("ANTHROPIC_API_KEY")

    #Create a client 
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system= "You only have the information given by the queries, you need to answer only based in this information and always give the source of your information",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    
    answer = message.content[0].text
    return answer

# ENTRY POINT: allows running this script directly from the terminal
if __name__ == "__main__":
    question = input("What do you want to know about RAG? ")
    answer = run_query(query=question, top_k=6)
    print("\n ANSWER \n")
    print(answer)