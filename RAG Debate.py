import openai
import os
import re
import pickle
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # OpenAI's tokenizer

# Function to estimate the token length of text
def estimate_tokens(text, model_name='gpt-4o-mini'):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_text, filename, api_key, max_tokens=4096):
    
    client = openai.OpenAI(
    api_key=api_key
    )

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
    
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                system_message,
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.1
        )

    generated_text = response.choices[0].message.content.strip()
    
    return generated_text, filename

# Function to save data to CSV
def save_to_csv(data, filename='chatbot_data.csv'):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filename']])

def save_preprocessed_data(chunked_paragraphs, vectors, vectorizer, filenames, file_prefix='preprocessed_data'):
    with open(f'{file_prefix}_chunks.pkl', 'wb') as file:
        pickle.dump(chunked_paragraphs, file)
    with open(f'{file_prefix}_vectors.pkl', 'wb') as file:
        pickle.dump(vectors, file)
    with open(f'{file_prefix}_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    with open(f'{file_prefix}_filenames.pkl', 'wb') as file:
        pickle.dump(filenames, file)


def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model('gpt-4')
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
    
# Chatbot function
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
        print(f"Context: {best_retrieved_text}")
        print(f"Response: {best_response}")

# Main function
def main():
    folder_path = 'archive/'
    
    # Replace with your OpenAI API key
    api_key = 'insert-your-api-key'
    
    chatbot_with_prevectorized_chunks(api_key)

if __name__ == "__main__":
    main()
