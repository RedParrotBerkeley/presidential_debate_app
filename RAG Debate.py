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

# Function to vectorize paragraphs
def vectorize_paragraphs(paragraphs):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(paragraphs)
    return vectorizer, vectors

# Function to search for the most similar paragraph
def search(query, vectorizer, vectors, paragraphs):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    idx = np.argmax(similarities)
    return paragraphs[idx], similarities[idx]

# Function to estimate the token length of text
def estimate_tokens(text, model_name='gpt-4'):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to truncate text to fit within a token limit
def truncate_text_to_fit(text, token_limit, model_name='gpt-4'):
    encoding = tiktoken.encoding_for_model(model_name)
    encoded = encoding.encode(text)
    return encoding.decode(encoded[:token_limit])

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_text, api_key, max_tokens=4096):
    openai.api_key = api_key
    print(f"Retrieved Text: {retrieved_text[:200]}...")  # Debug statement to show only the first 200 characters
    prompt = f"Context: {retrieved_text}\nQuestion: {query}\nAnswer:"
    
    # Adjust the prompt if it exceeds the maximum token length
    if estimate_tokens(prompt) > max_tokens:
        context_length = max_tokens - estimate_tokens(f"Question: {query}\nAnswer:")
        truncated_retrieved_text = truncate_text_to_fit(retrieved_text, context_length)
        prompt = f"Context: {truncated_retrieved_text}\nQuestion: {query}\nAnswer:"
    
    print(f"Prompt: {prompt[:200]}...")  # Debug statement to show only the first 200 characters
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are providing a summary limited to the content of the provided document as Donald Trump. Limit the summary to 200 characters. Do not mention Biden. Stick to the facts and avoid mannerisms, never refer to yourself as Donald Trump. Keep the answer short, clear, and concise. Speak as if you were answering the question, do not mention the document or documents referenced"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # Extract the relevant portion of the generated text
    generated_text = response.choices[0]['message']['content'].strip()
    print(f"Generated Response: {generated_text}")  # Debug statement
    return generated_text

# Function to save data to CSV
def save_to_csv(data, filename='chatbot_data.csv'):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filename']])

def save_preprocessed_data(paragraphs, filenames, file_prefix='preprocessed_data'):
    with open(f'{file_prefix}_paragraphs.pkl', 'wb') as file:
        pickle.dump(paragraphs, file)
    with open(f'{file_prefix}_filenames.pkl', 'wb') as file:
        pickle.dump(filenames, file)

def load_preprocessed_data(file_prefix='preprocessed_data'):
    with open(f'{file_prefix}_paragraphs.pkl', 'rb') as file:
        paragraphs = pickle.load(file)
    with open(f'{file_prefix}_filenames.pkl', 'rb') as file:
        filenames = pickle.load(file)
    return paragraphs, filenames

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

def chatbot(processed_paragraphs, all_filenames, api_key):
    chunked_paragraphs = chunk_paragraphs(processed_paragraphs, all_filenames, max_tokens=2000, model_name='gpt-4')
    print(f"Number of chunks: {len(chunked_paragraphs)}")  # Debug statement

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        best_chunk = None
        best_similarity = -1
        best_retrieved_text = None
        best_filenames = None
        
        for i, (chunk, filenames) in enumerate(chunked_paragraphs):
            if not chunk:
                continue
            vectorizer, vectors = vectorize_paragraphs(chunk)
            if vectors.shape[0] == 0:
                continue
            try:
                print(f"Processing chunk {i+1}/{len(chunked_paragraphs)} from files: {set(filenames)}")  # Debug statement
                retrieved_text, similarity = search(query, vectorizer, vectors, chunk)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_chunk = chunk
                    best_retrieved_text = retrieved_text
                    best_filenames = filenames

            except ValueError as e:
                print(f"Error processing chunk {i+1}/{len(chunked_paragraphs)}: {e}")  # Debug statement

        if best_retrieved_text:
            best_response = generate_response(query, best_retrieved_text, api_key, max_tokens=2000)
            print(f"Best response generated from chunk with files: {set(best_filenames)}")
        else:
            best_response = "No suitable chunk found."
            print("No suitable chunk found.")

        if best_response:
            data = {
                'query': query,
                'retrieved_text': best_retrieved_text,
                'response': best_response,
                'filename': ', '.join(best_filenames) if best_filenames else 'N/A'
            }
            save_to_csv(data)

def main():
    folder_path = 'Downloads/archive'
    preprocessed_data_prefix = 'preprocessed_data'
    
    try:
        # Load preprocessed data and filenames
        processed_paragraphs, all_filenames = load_preprocessed_data(preprocessed_data_prefix)
        print(f"Loaded {len(processed_paragraphs)} preprocessed paragraphs")  # Debug statement
    except FileNotFoundError:
        # Preprocess and save data if not already done
        all_texts = []
        all_filenames = []
        file_count = 0
        
        # List all files in the directory
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
        save_preprocessed_data(processed_paragraphs, all_filenames, preprocessed_data_prefix)
    
    # Replace with your OpenAI API key
    api_key = 'ENTER API HERE'

    chatbot(processed_paragraphs, all_filenames, api_key)

if __name__ == "__main__":
    main()
