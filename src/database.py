import sqlite3
import numpy as np
import logging
import os
import io
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DB_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'vectors.db')
DB_DIR = os.path.dirname(DB_FILE)

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

def get_db_connection():
    """Gets a connection to the SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the vectors table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    vector BLOB NOT NULL
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_collection_user ON vectors (collection, user_id);")
            conn.commit()
        logger.info(f"Database initialized successfully at {DB_FILE}")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def add_vector(collection: str, user_id: str, vector: np.ndarray) -> int:
    """
    Adds a vector for a user to a specific collection.
    The vector is stored as a BLOB.
    """
    if not isinstance(vector, np.ndarray) or vector.dtype != np.float32:
        raise TypeError("Vector must be a float32 numpy array.")
    
    vector_blob = vector.tobytes()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO vectors (collection, user_id, vector) VALUES (?, ?, ?)",
            (collection, user_id, vector_blob)
        )
        conn.commit()
        return cursor.lastrowid

def delete_vectors_by_user(collection: str, user_id: str) -> int:
    """Deletes all vectors for a given user_id in a collection."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM vectors WHERE collection = ? AND user_id = ?",
            (collection, user_id)
        )
        conn.commit()
        return cursor.rowcount

def search_vectors(collection: str, query_vector: np.ndarray, threshold: float, top_k: int) -> List[Dict[str, Any]]:
    """
    Searches for similar vectors in a collection using cosine similarity.
    Returns a list of dictionaries with 'user_id', 'similarity', and 'id'.
    """
    if not isinstance(query_vector, np.ndarray) or query_vector.dtype != np.float32:
        raise TypeError("Query vector must be a float32 numpy array.")

    # 1. Normalize the query vector
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector /= norm

    # 2. Fetch all vectors from the specified collection
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, user_id, vector FROM vectors WHERE collection = ?", (collection,))
        all_vectors_raw = cursor.fetchall()
    
    if not all_vectors_raw:
        return []

    # 3. Prepare for search
    db_vectors_list = [np.frombuffer(item['vector'], dtype=np.float32) for item in all_vectors_raw]
    db_vectors = np.array(db_vectors_list)
    
    # 4. Perform cosine similarity calculation (dot product of normalized vectors)
    # The vectors in the DB are assumed to be normalized already.
    similarities = np.dot(db_vectors, query_vector)
    
    # 5. Find matches above the threshold
    candidate_indices = np.where(similarities > threshold)[0]
    
    if len(candidate_indices) == 0:
        return []

    # 6. Sort candidates by similarity and select top_k
    top_indices = sorted(candidate_indices, key=lambda i: similarities[i], reverse=True)[:top_k]

    # 7. Format results
    results = [
        {
            "id": all_vectors_raw[i]['id'],
            "user_id": all_vectors_raw[i]['user_id'],
            "similarity": float(similarities[i])
        } for i in top_indices
    ]
    
    return results
