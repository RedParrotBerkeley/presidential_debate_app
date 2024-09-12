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

# Load environment variables from the .env file located in the /app directory
load_dotenv(dotenv_path="/app/.env")  # Correctly point to where .env is copied in the container

# Access the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI client
import openai
openai.api_key = openai_api_key
# Initialize FastAPI app
app = FastAPI()

# Include the router from endpoints
app.include_router(router)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
