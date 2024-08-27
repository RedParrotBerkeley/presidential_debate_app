import openai
import os
import pickle
import csv
import numpy as np
from scipy.spatial.distance import cosine
import tiktoken  # OpenAI's tokenizer
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# Set your OpenAI API key
# api_key = os.getenv('OPENAI_API_KEY', 'INSERT API')  
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model to use for chat completions
model = 'gpt-4o-mini'

client = OpenAI()
print("API client initialized successfully.")

# Function to estimate the token length of text
def estimate_tokens(text, model_name=model):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_texts, max_tokens=4096):
    
    client = openai.OpenAI()

    context = ', '.join(retrieved_texts)

    system_message = {
        "role": "system",
        "content": "Answer the question as if you are the speaker. Answer with facts based only on the provided context, and if the context does not have enough relevant information, say that you do not have enough information to answer the question. Avoid any mention of specific political parties or candidates. Keep the answer short, clear, and concise."
    }

    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    # Adjust the prompt if it exceeds the maximum token length
    if estimate_tokens(prompt) > max_tokens:
        context_length = max_tokens - estimate_tokens(f"Question: {query}\n\nAnswer:")
        truncated_retrieved_text = truncate_text_to_fit(context, context_length)
        prompt = f"Context: {truncated_retrieved_text}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = client.chat.completions.create(
                model= model,
                messages=[
                    system_message,
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                temperature=0.1
            )

        generated_text = response.choices[0].message.content.strip()
        return generated_text
    except Exception as e:
        print(f"Error generating chat completion: {e}")
        return None

def save_to_csv(data, filename='chatbot_data.tsv'):
    file_exists = os.path.isfile(filename)
    
    # Open the file with UTF-8 encoding
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filename']])

def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

   
def get_openai_embedding(text, model="text-embedding-3-small"):
    
    try:
        print("trying to embed ", text)
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
        
# Get the texts associated with the top n best similarity. Takes in a query embedding. Returns a dataframe
def find_best_texts(query_embedding, n):
    with open('vectorized_chunks.pkl', 'rb') as file:
        vectorized_chunks = pickle.load(file)

        best_retrieved_texts = []
        best_filenames = []
        best_similarities = []

        for embedding, chunk, filenames in vectorized_chunks:
            similarity_score = cosine(query_embedding, embedding)
            if similarity_score > 0:
                best_similarities.append(similarity_score)
                best_retrieved_texts.append(chunk[0]) 
                best_filenames.append(filenames[0])
        
        text_similarities = pd.DataFrame(
            {'texts': best_retrieved_texts,
            'filenames': best_filenames,
            'similarities': best_similarities
            })
            
        result = text_similarities.sort_values('similarities', ascending=False)
        print(result.head())
        file.close()
        return result.head(n)

# Chatbot function
def chatbot_with_prevectorized_chunks():
    
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        # Get the embedding for the user's query
        query_embedding = get_openai_embedding(query)

        if query_embedding is None:
            print("Failed to get query embedding. Try again.")
            continue
        best_texts_df = find_best_texts(query_embedding, 4)
        best_retrieved_texts = best_texts_df["texts"].tolist()
        best_filenames = best_texts_df["filenames"].tolist()

        if best_retrieved_texts:
            best_response = generate_response(query, best_retrieved_texts)
        else:
            best_response = "No suitable chunk found."
            best_filenames = 'N/A'

        if best_response:
            data = {
                'query': query,
                'retrieved_text': best_retrieved_texts,
                'response': best_response,
                'filename': best_filenames
            }
            save_to_csv(data)

        print(f"Response: {best_response}")

# Main function
def main():
    chatbot_with_prevectorized_chunks()

if __name__ == "__main__":
    main()
