from fastapi import FastAPI
from app.endpoints import router  # Ensure this import points to your endpoints file
from app.utils import (
    insert_into_database,
    select_from_database,
    generate_response,
    get_openai_embedding,
    find_best_texts,
    save_to_db,
)
from dotenv import load_dotenv
import os
import openai

# Load environment variables from the .env file located in the /app directory
load_dotenv(dotenv_path=".env")

# Access the API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Include the router from endpoints
app.include_router(router)

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Debate Bot API!"}
