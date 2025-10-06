try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop(
        'pysqlite3')  # Override default sqlite3
    import sqlite3
    print(f"SQLite version: {sqlite3.sqlite_version}")  # Should show >= 3.35.0
except ImportError:
    print("Error: pysqlite3-binary not installed. Run: pip install pysqlite3-binary")
    exit(1)

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
#Initialize ChromaDB Client
client = chromadb.Client()


def json_to_csv(json_data):
    return pd.DataFrame(json_data)

#Initialize Vector Database
def init_vectordb():
    dudedb = client.get_or_create_collection("dudes_collection")
    girldb = client.get_or_create_collection("girls_collection")
    return dudedb, girldb

#Read Files and Get Data
def read_files(dude_path, girl_path):
    pd.set_option('display.max_colwidth', None)
    dude_data = pd.read_csv(dude_path)
    girl_data = pd.read_csv(girl_path)
    return dude_data, girl_data

model = SentenceTransformer("all-MiniLM-L6-v2")

#Retrieve and Store Collection Data
def get_collection_data(file_data):
    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for person in file_data.itertuples():
        bio = person[2]
        name = person[1]

        embeddings.append(model.encode(bio).tolist())
        metadatas.append({'Name': name})
        ids.append(str(len(documents)))
        documents.append(bio)
    return documents, embeddings, metadatas, ids

#Insert Data to Vector Database
def insert_to_vectordb():
    dudedb, girldb = init_vectordb()
    dude_data, girl_data = read_files("dudes.csv", "girls.csv")

    dude_col = get_collection_data(dude_data)
    girl_col = get_collection_data(girl_data)
    dudedb.add(documents=dude_col[0], embeddings = dude_col[1], metadatas=dude_col[2], ids=dude_col[3])
    girldb.add(documents=girl_col[0], embeddings = girl_col[1], metadatas=girl_col[2], ids=girl_col[3])
    return 

#Search Vector Database for Results
def search_vectordb(name, n=1):
    dudedb, girldb = init_vectordb()
    dude_data, girl_data = read_files("dudes.csv", "girls.csv")
    
    if name in dude_data['name'].tolist():
        bio = dude_data[dude_data['name'] == name].iloc[0]['bio']
        results = girldb.query(query_texts=[bio], n_results=n)
    else:
        bio = girl_data[girl_data['name'] == name].iloc[0]['bio']
        results = dudedb.query(query_texts=[bio], n_results=n)
    return results

#Will match you up with other individual based on your bio
if __name__ == "__main__":
    insert_to_vectordb()
    name="Shayana Marie"
    #Raise n for more results if you have larger datasets 
    n=1 
    results = search_vectordb(name, n)
    print(results)