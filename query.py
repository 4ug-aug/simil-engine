import json
import logging
from typing import List

import chromadb
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_chroma_client(path: str = "chroma_db") -> chromadb.Client:
    """Creates a persistent Chroma client."""
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(client: chromadb.Client, name: str) -> chromadb.Collection:
    """Gets or creates a collection."""
    return client.get_or_create_collection(name=name)


class ChromaQuery(BaseModel):
    query_texts: List[str] = Field(description="The query to search for.")
    n_results: int = Field(description="The number of results to return.")


def query_chroma_client(
    collection: chromadb.Collection, query: ChromaQuery
) -> List[chromadb.Documents]:
    """Queries the Chroma client."""
    return collection.query(query_texts=query.query_texts, n_results=query.n_results)


def main():
    logger.info("Starting query pipeline...")
    client = create_chroma_client()
    collection = get_or_create_collection(client, "enron_emails")

    query = ChromaQuery(query_texts=["Hello, world!"], n_results=10)
    results = query_chroma_client(collection, query)
    logger.info(f"Results: {json.dumps(results, indent=4)}")


if __name__ == "__main__":
    main()
