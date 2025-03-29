import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#Step1 : Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"


llm=HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    temperature=0.5,
    model_kwargs={"token" : HF_TOKEN , "max_length":"512"}
)

#Step2: Connect LLM with FAISS and create Chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt

#LOAD Database
DB_FAISS_PATH="vectorstore/db_faiss"
def load_FAISS(DB_FAISS_PATH,embedding_model):
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError("Error: FAISS database not found at : {DB_FAISS_PATH}")
    return FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=load_FAISS(DB_FAISS_PATH,embedding_model)
# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])


# import os

# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from huggingface_hub import InferenceClient


# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm


# def load_llm(huggingface_repo_id):
#     client = InferenceClient(model=huggingface_repo_id, token=HF_TOKEN)
    
#     def query_model(question):
#         response = client.text_generation(question, max_new_tokens=512, temperature=0.5)
#         return response
    
#     return query_model
# # Step 2: Connect LLM with FAISS and Create chain

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load Database
# DB_FAISS_PATH="vectorstore/db_faiss"
# embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Create QA chain
# qa_chain=RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])

# import os
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from huggingface_hub import InferenceClient
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN=os.environ.get("HF_TOKEN")

# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN is missing! Check your .env file.")

# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Load FAISS Database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Hugging Face Inference Client
# client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# CUSTOM_PROMPT_TEMPLATE = """
# Use the provided context to answer the user's question.
# If you do not know the answer, say "I don't know." Do not make up information.
# Answer only based on the given context.

# Context: {context}
# Question: {question}

# Start the answer directly, no small talk.
# """

# def get_response(user_query):
#     retrieved_docs = db.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(user_query)
#     context = "\n".join([doc.page_content for doc in retrieved_docs])
    
#     # Format the prompt
#     prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=user_query)
    
#     # Query the LLM
#     response = client.text_generation(prompt, max_new_tokens=512, temperature=0.5)
    
#     return response, retrieved_docs

# # Invoke Query
# user_query = input("Write Your Query Here: ")
# response, source_docs = get_response(user_query)

# print("\nRESULT:", response)
# print("\nSOURCE DOCUMENTS:", [doc.page_content for doc in source_docs])
