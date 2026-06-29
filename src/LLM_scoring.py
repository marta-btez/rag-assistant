# FILE: LLM_scoring.py
# DESCRIPTION: LLM-as-a-judge: Load the results of the RAG model for a given query and analyzes bia LLM ans its knowledge the score of the answer
#              Returns the score btw 0-3
# AUTHOR: Marta Benitez Aguilar
# DATE: 2026-06-27

#IMPORTS
import os 
from dotenv import load_dotenv
import anthropic


#LOAD ENVIRONMENT VARIABLES 
load_dotenv()   #Reads the .env file and makes ANTHROPIC_API_KEY available in the environment

def run_LLM_scoring(query, answer, gold_answer):

    #CALL THE API

    #Read the API key from the environment
    api_key = os.getenv("ANTHROPIC_API_KEY")

    #Create a client 
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system= "You are an expert in AI and know by heart the five fundamental papers Attention is All you need, BERT, GPT-3, LoRA and RAG. You are evaluating a RAG model based in these 5 papers",
        messages=[
            {"role": "user", "content": f"Give a mark following this instructions: 0 - Incoherent or hallucination (factually incorrect) answer\
             1 - Unjustified out of scope (system claims no information exists when the chunk is present in the papers)\
             2 - Responds but incomplete (touches the concept but lacks detail, or does not cite the source correctly)\
             3 - Complete, detailed answer with correct source citation\
             if the question was {query}, the defined gold answer was {gold_answer} and the real answer of the RAG model is {answer}. \
            Return ONLY a single digit: 0, 1, 2 or 3. No explanation, no justification, no punctuation. Just the digit."}
        ]
    )

    
    LLM_score = message.content[0].text
    return int(LLM_score.strip())

