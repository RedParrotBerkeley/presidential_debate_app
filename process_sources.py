import openai
import os
import re
import pickle
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # OpenAI's tokenizer


# Function to extract text from a .txt file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().split('\n\n')  # Return a list of paragraphs separated by double newlines

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Function to preprocess paragraphs
def preprocess_paragraphs(paragraphs):
    return [preprocess_text(para) for para in paragraphs]


# Function to estimate the token length of text
def estimate_tokens(text, model_name='gpt-4'):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to chunk paragraphs into smaller pieces based on token length
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
            print(filenames)
            vectors = vectorizer.transform(chunk)
            if vectors.shape[0] > 0:
                vectorized_chunks.append((vectorizer, vectors, chunk, filenames))

    with open('vectorized_chunks.pkl', 'wb+') as file:
        pickle.dump(vectorized_chunks, file)
        file.close()

def main():
    folder_path = 'archive/'
    preprocess_and_vectorize_combined(folder_path)

if __name__ == "__main__":
    main()