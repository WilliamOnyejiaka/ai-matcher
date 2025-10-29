from typing import Generator, List, Tuple
from app.utils.reformat_to_bio import reformat_to_bio
from app.config.ai import model, index
from app.config.logger import logger


class AI:

    @classmethod
    def json_to_embeddings_bulk(cls, json_data: List[dict], batch_size: int = 100) -> Generator[Tuple[List, List], None, None]:
        for i in range(0, len(json_data), batch_size):
            batch = json_data[i:i + batch_size]
            bios = [reformat_to_bio(item)['bio'] for item in batch]
            try:
                batch_embeddings = model.encode(bios, batch_size=batch_size)
                ids = [item['_id'] for item in batch]
                yield batch_embeddings.tolist(), ids
            except Exception as e:
                logger.error(
                    f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                continue

    @classmethod
    def upsert_vectordb_bulk(cls, json_data: List[dict], batch_size: int = 100):
        successful = True
        for embeddings, ids in cls.json_to_embeddings(json_data, batch_size):
            try:
                vectors = [(id, emb, {"_id": str(id)})
                           for id, emb in zip(ids, embeddings)]
                index.upsert(vectors=vectors)
            except Exception as e:
                logger.error(f"Failed to insert batch to Pinecone: {str(e)}")
                successful = False
        return successful

    @classmethod
    def json_to_vector(cls, json_data: dict) -> List:
        result = reformat_to_bio(json_data)
        embeddings = model.encode(result['bio'])
        return [(result['_id'], embeddings, {"_id": str(result['_id'])})]

    @classmethod
    def upsert_vectordb(cls, vectors: List) -> bool:
        try:
            index.upsert(vectors=vectors)
            return True
        except Exception as e:
            logger.error(f"Failed to insert batch to Pinecone: {str(e)}")
            return False

    @classmethod
    def delete_embeddings(cls, ids: List):
        try:
            index.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to insert batch to Pinecone: {str(e)}")
            return False

    @classmethod
    def search_vectordb(cls, bio: str, n: int = 1, filter: dict = {}) -> List[str]:
        try:
            query_embedding = model.encode([bio])[0].tolist()
            results = index.query(
                vector=query_embedding,
                top_k=n,
                include_values=False,
                filter=filter
            )
            return [result.id for result in results.matches]
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
