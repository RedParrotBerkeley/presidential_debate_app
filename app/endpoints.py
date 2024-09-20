from fastapi import APIRouter, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
from typing import List, Dict
import secrets
from app.utils import (
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
    session_id: str

class ResponseModel(BaseModel):
    query_id: int
    responses: Dict[str, Dict[str, str]]

class SaveRequest(BaseModel):
    query_id: int
    candidate_id: int
    response: str
    retrieved_text: List[str]
    filenames: List[str]
    user_voted: int
    contexts: List[str]
    answer_relevancy: float
    faithfulness: float

# Start session endpoint
@router.get("/start-session/")
async def start_session(response: Response, request: Request):
    # Generate a session ID (or token)
    session_token = secrets.token_hex(16)
    
    # # Set the session ID in a cookie
    # response.set_cookie(key="session_id", value=session_token, httponly=True, secure=True, samesite='None')
    
     # Set 'secure' flag based on the request environment (True for HTTPS, False for localhost)
    #is_secure = "https" in request.url.scheme  # Check if the request is HTTPS
    #print(f"Is secure: {is_secure}: {request.url}")
    
    # Set the session ID in a cookie
    response.set_cookie(
        key="session_id",
        value=session_token,
        httponly=True,
        secure=True,  # Secure only for HTTPS
        samesite='None',
        domain=".ondigitalocean.app", 
        path="/"
    
    )

    # Optionally save session_token
    return {"message": "Session started", "session_id": session_token}

# Endpoint to receive user query, generate a response, and save to database
@router.post("/generate-response/", response_model=ResponseModel)
async def generate_response_endpoint(request: Request, req_body: QueryRequest):
    try:
        # Extract session_id from cookies
        print("Request cookies:", request.cookies)
        print("Request base URL", request.base_url)
        print("Request headers:", request.headers)
        print("Request url:", request.url)
        # commented out for now
        # session_id = request.cookies.get("session_id")
        session_id = req_body.session_id
        if not session_id:
            print("No session ID found in cookies")
            raise HTTPException(status_code=400, detail="Session ID is missing")
        print(f"Session ID received: {session_id}")

        # Extract query from the request body
        query = req_body.query
        if not query:
            raise HTTPException(status_code=400, detail="Query is missing")
        
        # Insert user query into the database
        vals = (session_id, query, datetime.now())  # Use session_id
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
            ['app/data/embeddings/vectorized_chunks_reichert.pkl'],  # List of .pkl filenames
            'sources/reichert',  # Folder path for .txt files
            4  # Number of best texts to retrieve
        )
        best_retrieved_texts_reichert = best_texts_df_reichert["texts"].tolist()
        source_url_reichert = best_texts_df_reichert["urls"].tolist()[0] if not best_texts_df_reichert.empty else "No URL found"

        # Generate a response for Reichert
        best_response_reichert = generate_response(query, best_retrieved_texts_reichert) if best_retrieved_texts_reichert else "No suitable chunk found for Reichert."

        # Retrieve texts for Ferguson
        best_texts_df_ferguson = find_best_texts(
            query_embedding,
            ['app/data/embeddings/vectorized_chunks_ferguson.pkl'],  # List of .pkl filenames
            'sources/ferguson',  # Folder path for .txt files
            4  # Number of best texts to retrieve
        )
        best_retrieved_texts_ferguson = best_texts_df_ferguson["texts"].tolist()
        source_url_ferguson = best_texts_df_ferguson["urls"].tolist()[0] if not best_texts_df_ferguson.empty else "No URL found"

        # Generate a response for Ferguson
        best_response_ferguson = generate_response(query, best_retrieved_texts_ferguson) if best_retrieved_texts_ferguson else "No suitable chunk found for Ferguson."

        # Flag non-answers from candidates
        if "i do not have" in best_response_reichert.lower() or "i do not have" in best_response_ferguson.lower():
            flag = {
                "query_id": query_id,
                "message": "One or both of these candidates have not discussed this topic, therefore we are unable to provide an answer at this time."
            }
            return JSONResponse(content=flag)
        
        # Prepare the dictionary response
        response_data_dict = {
            "query_id": query_id,
            "responses": {
                "reichert": {
                    "response": best_response_reichert,
                    "source_url": source_url_reichert,
                },
                "ferguson": {
                    "response": best_response_ferguson,
                    "source_url": source_url_ferguson,
                }
            }
        }

        # Save response to the database
        save_to_db({
            "query_id": query_id,
            "candidate_id": 1,  # Assuming "1" represents Reichert
            "response": best_response_reichert,
            "retrieved_text": best_retrieved_texts_reichert,
            "filenames": [txt for txt in best_texts_df_reichert["filenames"].tolist()],
            "user_voted": 0,  # Placeholder value for user voted
            "contexts": best_retrieved_texts_reichert,
            "answer_relevancy": 0.0,  # Placeholder value; replace with actual computation if needed
            "faithfulness": 0.0  # Placeholder value; replace with actual computation if needed
        })

        save_to_db({
            "query_id": query_id,
            "candidate_id": 2,  # Assuming "2" represents Ferguson
            "response": best_response_ferguson,
            "retrieved_text": best_retrieved_texts_ferguson,
            "filenames": [txt for txt in best_texts_df_ferguson["filenames"].tolist()],
            "user_voted": 0,  # Placeholder value for user voted
            "contexts": best_retrieved_texts_ferguson,
            "answer_relevancy": 0.0,  # Placeholder value; replace with actual computation if needed
            "faithfulness": 0.0  # Placeholder value; replace with actual computation if needed
        })

        # Return the dictionary as the response
        return response_data_dict

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())  # This will print the full traceback for better debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request. {str(e)}"
        )
