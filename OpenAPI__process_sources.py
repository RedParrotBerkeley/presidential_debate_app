import openai
import os
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # OpenAI's tokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model to use for embeddings
model = "text-embedding-3-small"

# Initialize OpenAI client (Note: OpenAI does not have a 'client' object; it's directly openai. Remove if unused)
openai.api_key = OPENAI_API_KEY
print("API client initialized successfully.")

def extract_text_and_url_from_txt(file_path):
    """
    Extracts text and URL from a .txt file. Assumes the URL is on the first line and the text follows.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        url = file.readline().strip()  # Read the URL from the first line
        paragraphs = file.read().split('\n\n')  # Read the paragraphs after the URL
    return url, paragraphs

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Function to preprocess paragraphs
def preprocess_paragraphs(paragraphs):
    return [preprocess_text(para) for para in paragraphs]

# Function to estimate the token length of text
def estimate_tokens(text, model_name=model):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def chunk_paragraphs(paragraphs, filenames, urls, chunk_size=2000, model_name=model):
    """
    Chunk paragraphs and keep track of filenames and URLs.
    """
    chunked_paragraphs = []
    current_chunk = []
    current_filenames = []
    current_urls = []
    
    for paragraph, filename, url in zip(paragraphs, filenames, urls):
        paragraph_length = len(paragraph)
        current_filenames = [filename]
        current_urls = [url]
        while paragraph_length > chunk_size:
            current_chunk = [paragraph[:chunk_size]]
            chunked_paragraphs.append((current_chunk, current_filenames, current_urls))
            paragraph = paragraph[chunk_size:]
            paragraph_length = len(paragraph)
        if (paragraph_length > 0) and (paragraph_length < chunk_size):
            chunked_paragraphs.append(([paragraph], [filename], [url]))
    
    return chunked_paragraphs

def get_openai_embedding(text, model=model):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input = [text], model=model).data[0].embedding
    
# Function to preprocess, chunk, and vectorize using OpenAI embeddings
def preprocess_and_vectorize_combined(folder_path, output_filename, chunk_size=2000, model_name=model):
    all_texts = []
    all_filenames = []
    all_urls = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            url, raw_paragraphs = extract_text_and_url_from_txt(file_path)
            all_texts.extend(raw_paragraphs)
            all_filenames.extend([filename] * len(raw_paragraphs))
            all_urls.extend([url] * len(raw_paragraphs))  # Store URL for each paragraph

    print(f"Extracted and preprocessed text from {len(all_filenames)} files in {folder_path}.")

    processed_paragraphs = preprocess_paragraphs(all_texts)
    chunked_paragraphs = chunk_paragraphs(processed_paragraphs, all_filenames, all_urls, chunk_size, model_name)

    vectorized_chunks = []
    for chunk, filenames, urls in chunked_paragraphs:
        if chunk:
            embedding = get_openai_embedding(chunk[0])  # Get embedding for the chunk
            if embedding:
                vectorized_chunks.append((embedding, chunk, filenames, urls))  # Include URLs
                print(f"Vectorized chunk from file {filenames[0]} with URL {urls[0]}.")

    with open(output_filename, 'wb+') as file:
        pickle.dump(vectorized_chunks, file)
    print(f"Vectorized chunks saved to {output_filename} successfully.")

def main():
    # Define folders and output filenames
    folder_paths = {
        'reichert': 'Downloads/sources/reichert/',
        'ferguson': 'Downloads/sources/ferguson/'
    }
    
    # Process both folders
    for candidate, folder_path in folder_paths.items():
        output_filename = f'vectorized_chunks_{candidate}.pkl'
        preprocess_and_vectorize_combined(folder_path, output_filename, chunk_size=500)
    
    print("Process completed successfully for both candidates.")

if __name__ == "__main__":
    main()
