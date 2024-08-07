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

# Preprocess, chunk, and vectorize in one step
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

# Function to generate a response using the ChatGPT model - continue with prompt engineering
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

# Function to save data to CSV - work in progress - needs refinement
def save_to_csv(data, filename='chatbot_data.csv'):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filename']])

# Chatbot function modified to handle preprocess_and_vectorize_combined
def chatbot_with_prevectorized_chunks(api_key):
    with open('vectorized_chunks.pkl', 'rb') as file:
        vectorized_chunks = pickle.load(file)

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

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

        if best_response:
            data = {
                'query': query,
                'retrieved_text': best_retrieved_text,
                'response': best_response,
                'filename': best_filename
            }
            save_to_csv(data)

        print(f"Filename: {best_filename}")
        print(f"Response: {best_response}")

# Main function
def main():
    folder_path = 'Downloads/archive'
    
    # Preprocess and save data
    all_texts = []
    all_filenames = []
    file_count = 0
    
    # List all files in the directory - part of my debugging efforts - can be removed later
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_count += 1
            file_path = os.path.join(folder_path, filename)
            raw_paragraphs = extract_text_from_txt(file_path)
            print(f"Processing file: {filename} with {len(raw_paragraphs)} paragraphs")  # Debug statement
            all_texts.extend(raw_paragraphs)
            all_filenames.extend([filename] * len(raw_paragraphs))
    
    processed_paragraphs = preprocess_paragraphs(all_texts)
    print(f"Processed {len(processed_paragraphs)} paragraphs from {file_count} files")  # Debug statement
    
    # Save the paragraphs and filenames
    save_preprocessed_data(processed_paragraphs, all_filenames)
    
    # Vectorize the preprocessed paragraphs
    preprocess_and_vectorize(folder_path)
    
    # Replace with your OpenAI API key
    api_key = 'add api'

    chatbot_with_prevectorized_chunks(api_key)

if __name__ == "__main__":
    main()
