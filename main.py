"""
FastAPI Application Flow:

1. **Imports and Setup**
    - Import necessary libraries and modules.
    - Define FastAPI application instance.
    - Set the OpenAI API key.

2. **Data Model**
    - Define a Pydantic model for request validation (`QueryRequest`).

3. **Utility Functions**
    - `extract_text_from_txt`: Extracts text from a .txt file.
    - `preprocess_text`: Preprocesses text by removing extra spaces and converting to lowercase.
    - `preprocess_paragraphs`: Applies text preprocessing to a list of paragraphs.
    - `estimate_tokens`: Estimates the number of tokens in a text for a given model.
    - `chunk_paragraphs`: Chunks paragraphs into smaller pieces based on token length.
    - `preprocess_and_vectorize_combined`: Preprocesses and vectorizes text files in a directory.
    - `truncate_text_to_fit`: Truncates text to fit within a specified token limit.
    - `generate_response`: Generates a response using OpenAI's ChatGPT model.
    - `save_to_csv`: Saves query and response data to a CSV file.

4. **Startup Event**
    - `load_data`: Preprocesses and vectorizes data when the app starts.

5. **Endpoint**
    - `query_handler`: Handles POST requests to the `/query` endpoint, processes the query, and returns a response.

6. **Main Function**
    - Runs the FastAPI application using Uvicorn.

---
*HOW TO USE*
---
1. From command prompt enter 
```sh
uvicorn main:app --reload
```
2. Copy URL and paste it into browser. 

3. Open another prompt and use a curl command to make a post to the server. 
```sh
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"what are your thoughts on immigration?\"}"
```
3(Alternate). Copy URL and paste it into the browser. After the URL add "/docs" and hit enter for a UI. 
"""

from fastapi import FastAPI, HTTPException
import openai
import os
import re
import pickle
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # OpenAI's tokenizer
from pydantic import BaseModel

api_key = 'insertAPI'
app = FastAPI()

#Define a Pydantic model for the request body. This model will validate the incoming request data.
class QueryRequest(BaseModel):
    query: str

#Include all the utility functions from your script. Ensure they are correctly defined within the FastAPI app context:
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().split('\n\n')  # Return a list of paragraphs separated by double newlines

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def preprocess_paragraphs(paragraphs):
    return [preprocess_text(para) for para in paragraphs]

def estimate_tokens(text, model_name='gpt-4'):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def chunk_paragraphs(paragraphs, filenames, max_tokens=2000, model_name='gpt-4'):
    chunked_paragraphs = []
    current_chunk = []
    current_length = 0
    current_filenames = []
    
    for paragraph, filename in zip(paragraphs, filenames):
        paragraph_length = estimate_tokens(paragraph, model_name)
        if current_length + paragraph_length > max_tokens:
            chunked_paragraphs.append((current_chunk, current_filenames))
            current_chunk = [paragraph]
            current_length = paragraph_length
            current_filenames = [filename]
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length
            current_filenames.append(filename)
    
    if current_chunk:
        chunked_paragraphs.append((current_chunk, current_filenames))
    
    return chunked_paragraphs

def preprocess_and_vectorize_combined(folder_path, max_tokens=2000, model_name='gpt-4'):
    all_texts = []
    all_filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            raw_paragraphs = extract_text_from_txt(file_path)
            all_texts.extend(raw_paragraphs)
            all_filenames.extend([filename] * len(raw_paragraphs))

    processed_paragraphs = preprocess_paragraphs(all_texts)
    chunked_paragraphs = chunk_paragraphs(processed_paragraphs, all_filenames, max_tokens, model_name)

    vectorizer = TfidfVectorizer()
    all_chunks = [chunk for chunk, filenames in chunked_paragraphs]
    flattened_chunks = [paragraph for chunk in all_chunks for paragraph in chunk if paragraph.strip()]

    if flattened_chunks:
        vectorizer.fit(flattened_chunks)

    vectorized_chunks = []
    for chunk, filenames in chunked_paragraphs:
        if chunk:
            vectors = vectorizer.transform(chunk)
            if vectors.shape[0] > 0:
                vectorized_chunks.append((vectorizer, vectors, chunk, filenames))

    with open('vectorized_chunks.pkl', 'wb') as file:
        pickle.dump(vectorized_chunks, file)


def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model('gpt-4')
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def generate_response(query, retrieved_text, filename, api_key, max_tokens=4096):
    openai.api_key = api_key

    system_message = {
        "role": "system",
        "content": "You are a neutral and monotone responder. Do not ever mention the speaker. Answer questions based only on the provided context. Avoid any mention of other parties, candidates, or specific individuals. Stick to the facts, avoid any mannerisms or opinions, and keep the answer short, clear, and concise. Respond directly as if you are the speaker."
    }

    prompt = f"Context: {retrieved_text}\n\nQuestion: {query}\n\nAnswer:"

    # Adjust the prompt if it exceeds the maximum token length
    if estimate_tokens(prompt) > max_tokens:
        context_length = max_tokens - estimate_tokens(f"Question: {query}\n\nAnswer:")
        truncated_retrieved_text = truncate_text_to_fit(retrieved_text, context_length)
        prompt = f"Context: {truncated_retrieved_text}\n\nQuestion: {query}\n\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            system_message,
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    generated_text = response.choices[0]['message']['content'].strip()
    return generated_text, filename

def save_to_csv(data, filename='chatbot_data.csv'):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filename']])

#FastAPI's 'on_event' decorator to load preprocessed and vectorized data when the app starts
@app.on_event("startup")
def load_data():
    folder_path = 'Downloads/archive'
    preprocess_and_vectorize_combined(folder_path)
    global vectorized_chunks
    with open('vectorized_chunks.pkl', 'rb') as file:
        vectorized_chunks = pickle.load(file)

#Create an endpoint that handles POST requests with the user's query. This endpoint will use the utility functions to process the query and return a response.
@app.post("/query")
async def query_handler(request: QueryRequest):
    query = request.query

    with open('vectorized_chunks.pkl', 'rb') as file:
        vectorized_chunks = pickle.load(file)

    best_similarity = -1
    best_retrieved_text = None
    best_filename = None

    for vectorizer, vectors, chunk, filenames in vectorized_chunks:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, vectors).flatten()
        idx = np.argmax(similarities)

        if similarities[idx] > best_similarity:
            best_similarity = similarities[idx]
            best_retrieved_text = chunk[idx]
            best_filename = filenames[idx]

    if best_retrieved_text:
        best_response, best_filename = generate_response(query, best_retrieved_text, best_filename, api_key, max_tokens=2000)
    else:
        best_response = "No suitable chunk found."
        best_filename = 'N/A'

    data = {
        'query': query,
        'retrieved_text': best_retrieved_text,
        'response': best_response,
        'filename': best_filename
    }
    save_to_csv(data)

    return {"response": best_response, "filename": best_filename}

if __name__ == "__main__":
    api_key = 'insertAPI'
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
