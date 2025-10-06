try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop(
        'pysqlite3')  # Override default sqlite3
    import sqlite3
except ImportError:
    print("Error: pysqlite3-binary not installed. Run: pip install pysqlite3-binary")
    exit(1)

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from bio import reformat_to_bio
from data import user, data
        

client = chromadb.Client()


def json_to_csv(json_data):
    return pd.DataFrame(json_data)


match_collection = "possible_matches_collection"


def init_vectordb(collection_name: str):
    return client.get_or_create_collection(collection_name)


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection_data(file_data):
    embeddings = []
    ids = []

    for _, row in file_data.iterrows():
        bio = row['bio']
        id = row['_id']

        embeddings.append(model.encode(bio).tolist())
        ids.append(id)

    return embeddings, ids


def insert_to_vectordb():
    possible_matches_json = []
    for item in data:
        possible_matches_json.append(reformat_to_bio(item))

    possible_matches_db = init_vectordb(
        match_collection)
    possible_matches_data = json_to_csv(
        possible_matches_json)

    possible_matches_col = get_collection_data(possible_matches_data)
    possible_matches_db.add(
        embeddings=possible_matches_col[0],
        ids=possible_matches_col[1]
    )
    return


def search_vectordb(bio: str, n=1):
    insert_to_vectordb()
    possible_matches_db = init_vectordb(match_collection)
    results = possible_matches_db.query(query_texts=[bio], n_results=n)

    return results['ids'][0]


if __name__ == "__main__":
    n = 100
    bio = reformat_to_bio(user)['bio']
    results = search_vectordb(bio, n)
    print(results)
