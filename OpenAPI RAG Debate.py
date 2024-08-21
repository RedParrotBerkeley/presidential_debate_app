import openai
import os
import pickle
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # OpenAI's tokenizer
from openapi_client.api.default_api import DefaultApi  # Import the API client
from openapi_client.models.generate_chat_completion_request import GenerateChatCompletionRequest  # Import the request model
from openapi_client.configuration import Configuration  # Import the configuration class
from openapi_client.api_client import ApiClient  # Import the ApiClient class

#Insert API on lines 16,66, and 100 having issues calling api_key in some instances 

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY', 'INSERT API')

# Model to use for chat completions
model = 'gpt-4o-mini'

# Configure API client with the API key in the headers
configuration = Configuration()
configuration.api_key_prefix['Authorization'] = 'Bearer'
configuration.api_key['Authorization'] = api_key

# Initialize the API client with the configuration
api_client = ApiClient(configuration=configuration)

# Initialize the DefaultApi client
client = DefaultApi(api_client=api_client)

# Function to estimate the token length of text
def estimate_tokens(text, model_name=model):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_texts, filenames, max_tokens=4096):
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

    request_body = GenerateChatCompletionRequest(
        model=model,
        messages=[
            system_message,
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        temperature=0.1
    )

    headers = {
        "Authorization": f"Bearer {'INSERT API'}"
    }

    try:
        response = client.generate_chat_completion(request_body, _headers=headers)
        generated_text = response.choices[0].message.content.strip()
        return generated_text, filenames
    except Exception as e:
        print(f"Error generating chat completion: {e}")
        return None, filenames

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

def get_openai_embedding(text):
    try:
        # Ensure you're using the embedding model, not the chat model
        request_body = GenerateEmbeddingsRequest(input=[text], model='text-embedding-ada-002')
        headers = {
            "Authorization": f"Bearer {'INSERT API'}"
        }
        print(f"Request Body: {request_body}")
        print(f"Headers: {headers}")
        
        response = client.generate_embeddings(request_body, _headers=headers)
        print(f"Embedding generated successfully for text: {text[:30]}...")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
        
# Chatbot function
def chatbot_with_prevectorized_chunks(min_similarity=.15):
    with open('vectorized_chunks.pkl', 'rb') as file:
        vectorized_chunks = pickle.load(file)

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        # Get the embedding for the user's query
        query_embedding = get_openai_embedding(query)

        if query_embedding is None:
            print("Failed to get query embedding. Try again.")
            continue

        best_retrieved_texts = []
        best_filenames = []

        for embedding, chunk, filenames in vectorized_chunks:
            similarities = cosine_similarity([query_embedding], [embedding]).flatten()
            idx = np.argmax(similarities)

            if similarities[idx] > min_similarity:
                print(f"Similarity: {similarities[idx]}")
                best_retrieved_texts.append(chunk[0])  # Get the text associated with the best similarity
                best_filenames.append(filenames[0])

        if best_retrieved_texts:
            best_response, best_filenames = generate_response(query, best_retrieved_texts, best_filenames)
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

        #print(f"Filename(s): {best_filenames}")
        #print(f"Context: {best_retrieved_texts}")
        print(f"Response: {best_response}")

# Main function
def main():
    chatbot_with_prevectorized_chunks(min_similarity=.80)

if __name__ == "__main__":
    main()
