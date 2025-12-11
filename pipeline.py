import logging
import re
from pathlib import Path
from typing import List

import chromadb
import mailparser
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "emails.csv"
OUTPUT_FILE = DATA_DIR / "emails_cleaned.csv"


def create_chroma_client(path: str = "chroma_db") -> chromadb.Client:
    """Creates a persistent Chroma client."""
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(
    chroma_client: chromadb.Client, name: str
) -> chromadb.Collection:
    """Gets or creates a collection."""
    return chroma_client.get_or_create_collection(name=name)


class ChromaEmailEntries(BaseModel):
    ids: List[str] = Field(description="The ids of the email.")
    documents: List[str] = Field(description="The documents of the email.")
    metadatas: List[dict] = Field(description="The metadatas of the email.")

    @field_validator("documents", mode="before")
    def validate_documents(cls, v: list) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("Documents must be a list.")
        return [str(item) if isinstance(item, str) else str(item) for item in v]

    @field_validator("ids", mode="before")
    def validate_ids(cls, v: list) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("Ids must be a list.")
        return [str(item) if isinstance(item, str) else str(item) for item in v]


def add_entries_to_collection(
    collection: chromadb.Collection, entries: ChromaEmailEntries
) -> bool:
    """Add entries to the collection."""
    try:
        collection.upsert(
            ids=entries.ids,
            documents=entries.documents,
            metadatas=entries.metadatas,
        )
    except Exception as e:
        logger.error(f"Failed to add entries to collection: {e}")
        return False
    return True


def clean_text(text: str) -> str:
    """Normalizes whitespace in text."""
    if not isinstance(text, str):
        return ""
    # Replaces newlines, tabs, and multiple spaces with a single space
    return re.sub(r"\s+", " ", text).strip()


def parse_email_content(raw_message: str) -> pd.Series:
    """
    Parses a raw email string and extracts relevant fields.
    Returns a Pandas Series to allow easy column assignment.
    """
    try:
        email = mailparser.parse_from_string(raw_message)
        return pd.Series(
            {
                "cleaned_body": clean_text(email.body),
                "subject": email.subject,
                "from": email.from_[0][1],
                "to": email.to[0][1],
                "date": email.date.strftime("%Y-%m-%d %H:%M:%S"),
                "message_id": email.message_id,
                "in_reply_to": email.in_reply_to,
                "references": email.references,
            }
        )
    except Exception as e:
        logger.error(f"Failed to parse email: {e}")
        return pd.Series(dtype="object")


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Preprocessing {len(data)} emails...")

    extracted_data = data["message"].apply(parse_email_content)

    processed_df = pd.concat([data, extracted_data], axis=1)

    return processed_df


def main():
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    logger.info(f"Loading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE).iloc[:100]

    df_cleaned = preprocess_data(df)

    logger.info(
        f"Processed Data Head:\n{df_cleaned[['subject', 'cleaned_body']].head()}"
    )

    logger.info(f"Saving data to {OUTPUT_FILE}")
    df_cleaned.to_csv(OUTPUT_FILE, index=False)

    chroma_client = create_chroma_client()
    collection = get_or_create_collection(chroma_client, "enron_emails")
    entries = ChromaEmailEntries(
        ids=df_cleaned["message_id"].tolist(),
        documents=df_cleaned["cleaned_body"].tolist(),
        metadatas=df_cleaned[["subject", "from", "to", "date"]].to_dict(
            orient="records"
        ),
    )
    add_entries_to_collection(collection, entries)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
