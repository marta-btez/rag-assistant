# FILE: semantic_scoring.py
# DESCRIPTION: cosine similarity: using embeddings, returns the % of similarity btw the gold answer and the real answer 
# AUTHOR: Marta Benitez Aguilar
# DATE: 2026-06-27

#IMPORTS
from langchain_huggingface import HuggingFaceEmbeddings   #To convert text to vectors (multiple options but HF free, OpenAI is another possibility but not free)
from sklearn.metrics.pairwise import cosine_similarity

def run_semantic_scoring(answer, gold_answer):

    #CHARGE THE EMBEDDINGS MODEL
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"   #SAME AS THE INDEXER, but here il could be different 
    )
    # CONVERT BOTH TEXTS TO VECTORS
    vectors = embedding_model.embed_documents([answer, gold_answer])  #returns a list of 2 vectors
    answer_vector = vectors[0]   #first vector corresponds to answer
    gold_vector = vectors[1]     #second vector corresponds to gold_answer

    # CALCULATE COSINE SIMILARITY BETWEEN THE TWO VECTORS
    similarity = cosine_similarity([answer_vector], [gold_vector])  #expects 2D arrays
    similarity_score = float(similarity[0][0])  #extract the single value from the result matrix, returns a ratio 0-1, being 1 identic vectors 

    return similarity_score