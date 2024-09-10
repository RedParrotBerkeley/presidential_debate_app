from fastapi import FastAPI
from endpoints import router  # Ensure this import points to your endpoints file
from utils import (
    insert_into_database,
    select_from_database,
    generate_response,
    get_openai_embedding,
    find_best_texts,
    save_to_db,
)

# Initialize FastAPI app
app = FastAPI()

# Include the router from endpoints
app.include_router(router)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)