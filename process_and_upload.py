import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import *

df = pd.read_parquet(DATA_PATH)
print('Dataset Read Successfully...')

df = df.fillna("")
print('Removed None, Null , empty values...')
OPTIONS = {
	0 : "opa",
	1 : "opb",
	2 : "opc",
	3 : "opd"
}

def get_answer(x):
	answer = x[OPTIONS.get(x['cop'], "Cannot find Answer")]
	return answer

def merge_data(row):
	merged_data = f"""Question: {row["question"]},
Answer: {row['answer']}
Subject-Topic: {row["subject_name"] + row["topic_name"]}
Explanation:{row["exp"]}"""
	return merged_data

print('Started Merging...')
df['options'] = df.apply(lambda x : ", ".join([x[i] for i in OPTIONS.values()]), axis = 1)
df['answer'] = df.apply(lambda x : get_answer(x), axis= 1)
df['data'] = df.apply(lambda x : merge_data(x), axis= 1)
df = df[['id','data','question','options', 'answer']]
print('Merging Completed.')

print('Downloading Embeddings...')
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL)
print('Embeddings Downloaded')
vector_store = Chroma(
    collection_name= DB_COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_URI,  # Where to save data locally, remove if not necessary
)
print("Embeddings and Vector Store Loaded Successfully...")

test_df = df.iloc[:LIMIT]

def get_documents(df):
    documents = [Document(
        page_content=i['data'], 
        metadata={
			"question": i['question'],
			"options": i['options'],
			"answer": i['answer'],
		})
        for _, i in df.iterrows()
	]

    return documents
print('Uploading to VectorStore...')
vector_store.add_documents(documents=get_documents(test_df), ids=test_df['id'].tolist())
print('Data Uploaded Successfully...')
