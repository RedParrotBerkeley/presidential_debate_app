from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.utils import (  # Import from utils.py specifically
    insert_into_database,
    select_from_database,
    generate_response,
    get_openai_embedding,
    find_best_texts,
    save_to_db,
)
from datetime import datetime 

# Initialize FastAPI router
router = APIRouter()

# Pydantic models for request and response validation
class QueryRequest(BaseModel):
    query: str

class ResponseData(BaseModel):
    response: str
    source_url: str

class SaveRequest(BaseModel):
    query_id: int
    candidate_id: int
    response: str
    contexts: List[str]
    filenames: List[str]
    user_voted: int
    answer_relevancy_score: float
    faithfulness_score: float

# Endpoint to receive user query and generate a response
@router.post("/generate-response/", response_model=List[ResponseData])
async def generate_response_endpoint(request: QueryRequest):
    try:
        # Extract query from request - return response include query ID instead of array maybe put dictionary?
        query = request.query

        # Insert user query into the database
        vals = (0, query, datetime.now())  # Use appropriate session_id (replace 0)
        insert_into_database("INSERT INTO Query (sessionId, query, timestamp) VALUES (%s, %s, %s)", vals)

        # Retrieve the last query from the database
        query_id, query = select_from_database("SELECT id, query FROM Query ORDER BY id DESC LIMIT 1")[0]

        # Generate embedding for the user's query
        query_embedding = get_openai_embedding(query)

        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        # Retrieve texts for Reichert
        best_texts_df_reichert = find_best_texts(
            query_embedding,
            ['data/embeddings/vectorized_chunks_reichert.pkl'],  # List of .pkl filenames
            'sources/reichert',  # Folder path for .txt files
            4  # Number of best texts to retrieve
        )
        best_retrieved_texts_reichert = best_texts_df_reichert["texts"].tolist()
        source_url_reichert = best_texts_df_reichert["urls"].tolist()[0] if not best_texts_df_reichert.empty else "No URL found"

        # Generate a response for Reichert
        best_response_reichert = generate_response(query, best_retrieved_texts_reichert) if best_retrieved_texts_reichert else "No suitable chunk found for Reichert."

        # Print formatted response for Reichert
        print(f"Response for Reichert: {best_response_reichert}\nSource URL: {source_url_reichert}")

        # Retrieve texts for Ferguson
        best_texts_df_ferguson = find_best_texts(
            query_embedding,
            ['data/embeddings/vectorized_chunks_ferguson.pkl'],  # List of .pkl filenames
            'sources/ferguson',  # Folder path for .txt files
            4  # Number of best texts to retrieve
        )
        best_retrieved_texts_ferguson = best_texts_df_ferguson["texts"].tolist()
        source_url_ferguson = best_texts_df_ferguson["urls"].tolist()[0] if not best_texts_df_ferguson.empty else "No URL found"

        # Generate a response for Ferguson
        best_response_ferguson = generate_response(query, best_retrieved_texts_ferguson) if best_retrieved_texts_ferguson else "No suitable chunk found for Ferguson."

        # Print formatted response for Ferguson
        print(f"Response for Ferguson: {best_response_ferguson}\nSource URL: {source_url_ferguson}")

        # Return response as a dictionary
        return {
            "query_id": query_id,
            "responses": {
                "reichert": {
                    "response": best_response_reichert,
                    "source_url": source_url_reichert
                },
                "ferguson": {
                    "response": best_response_ferguson,
                    "source_url": source_url_ferguson
                }
            }
        }

    except Exception as e:
        # Improved exception handling
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again later."
        )

# Endpoint to save response data to the database
@router.post("/save-response/")
async def save_response(request: SaveRequest):
    try:
        # Save data to the database
        data = request.dict()
        save_to_db(data)
        return {"status": "Success", "message": "Data saved successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



