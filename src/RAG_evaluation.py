# FILE: RAG_evaluation.py
# DESCRIPTION: Chunking experiment for the AI Papers RAG Assistant.
#              Tests 9 configurations (3 CHUNK_SIZE x 3 TOP_K) against
#              a 10-question evaluation set and saves results to a file.
# AUTHOR: Marta Benitez Aguilar
# DATE: 2026-06-27

#IMPORTS
import os 
from indexer import run_indexer
from query import run_query
from LLM_scoring import run_LLM_scoring
from semantic_scoring import run_semantic_scoring
import pandas as pd  
import json

#RAG CONFIGURATION
CHUNK_SIZE_eval = [200, 500, 1000] #[500]
TOP_K_eval = [3, 5, 7] #[3] 
CHUNK_OVERLAP = 50 

#EVAL QUERY DATASET
with open("src/evaluation_set.json", "r") as f:
    QUERY_eval = json.load(f)

#RESULTS STORAGE
results = []  #empty list to store all results before aggregating

for chunk_size in CHUNK_SIZE_eval:
    run_indexer(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)

    for top_k in TOP_K_eval:
         
        for item in QUERY_eval:
            question = item["question"]        #the question string
            gold_answer = item["gold_standard"] #the gold standard string

            answer = run_query(query=question, top_k=top_k)

            LLM_score = run_LLM_scoring(query=question, answer=answer, gold_answer=gold_answer)

            semantic_score = run_semantic_scoring(answer=answer, gold_answer=gold_answer)
            results.append({         
                            "chunk_size": chunk_size,
                            "top_k": top_k,
                            "question": question,
                            "answer": answer,
                            "gold_answer": gold_answer,
                            "llm_score": LLM_score,
                            "semantic_score": semantic_score
                        })
            #break #test with one 
            
# AGGREGATE RESULTS BY CONFIGURATION
# Convert results list to a pandas DataFrame
df = pd.DataFrame(results)  # each row is one question + one configuration

# PIVOT TABLE: rows = configurations, columns = questions (LLM scores)
llm_table = df.pivot_table(
    index=["chunk_size", "top_k"],   # one row per configuration
    columns="question",              # one column per question
    values="llm_score"               # LLM score as values
)

# PIVOT TABLE: rows = configurations, columns = questions (semantic scores)
semantic_table = df.pivot_table(
    index=["chunk_size", "top_k"],   # one row per configuration
    columns="question",              # one column per question
    values="semantic_score"          # semantic score as values
)

# ADD TOTAL COLUMN
llm_table["TOTAL /30"] = llm_table.sum(axis=1)        # sum all LLM scores per row
semantic_table["TOTAL_AVG"] = semantic_table.mean(axis=1)  # average semantic score per row

# SAVE TO FILE
llm_table.to_csv("results_LLM_scoring.csv")           # save LLM table as CSV
semantic_table.to_csv("results_semantic_scoring.csv")  # save semantic table as CSV

print("Experiment complete. Results saved.")

    

    
