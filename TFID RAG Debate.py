import openai
import os
import re
import pickle
import csv
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # OpenAI's tokenizer
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime


#insert desired model 
model = 'gpt-4o-mini'

#load envrionment variables
load_dotenv()
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to create a database connection and run a query
def insert_into_database(sql_string, vals):
    print(sql_string)
    connection = mysql.connector.connect(user=MYSQL_USER, password=MYSQL_PASSWORD,
                                host=MYSQL_HOST,
                                port=MYSQL_PORT,
                                database=MYSQL_DATABASE)

    cursor = connection.cursor()
    cursor.execute(sql_string, vals)
    connection.commit()

    print(cursor.rowcount, "record inserted.")
    connection.close()

def select_from_database(sql_string):
    connection = mysql.connector.connect(user=MYSQL_USER, password=MYSQL_PASSWORD,
                                host=MYSQL_HOST,
                                port=MYSQL_PORT,
                                database=MYSQL_DATABASE)

    cursor = connection.cursor()
    cursor.execute(sql_string)
    records = cursor.fetchall()
    connection.close()
    return records

def get_last_query_from_db():
    rows = select_from_database("SELECT id, query FROM Query ORDER BY id desc LIMIT 1")
    print("rows:", rows)
    query_id = rows[0][0]
    print(query_id)
    query = rows[0][1]
    print(query)
    return (query_id, query)

# Function to estimate the token length of text
def estimate_tokens(text, model_name=model):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_texts, filenames, api_key, max_tokens=4096):
    
    client = openai.OpenAI(
    api_key=api_key
    )

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
    
    return generated_text, filenames

# Function to save data to CSV
def save_to_csv(data, filename='chatbot_data.tsv'):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filenames']])

def save_to_db(data):
    #TODO make this insert work when constrained foreign key values are set up
    contexts = json.dumps(data['retrieved_text'])
    filenames = json.dumps(data['filenames'])
    vals = (data['query_id'], 1, data['response'], contexts, filenames, 0, 0, 0) #TODO fill in missing fields
    insert_into_database(f"INSERT INTO Response (queryId, candidateId, response, contexts, filenames, userVoted, contextRelevanceScore, faithfulnessScore) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", vals)

def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
    
# Chatbot function
def chatbot_with_prevectorized_chunks(api_key, min_similarity=.15):
    with open('vectorized_chunks.pkl', 'rb') as file:
        vectorized_chunks = pickle.load(file)
    session_id = 0 #TODO replace this with real session
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        vals = (session_id, query, datetime.now())
        insert_into_database(f"INSERT INTO Query (sessionId, query, timestamp) VALUES (%s, %s, %s)", vals)
        #TODO move the insert statement to the front end

        best_retrieved_texts = []
        best_filenames = []

        query_id, query = get_last_query_from_db()
        
        for vectorizer, vectors, chunk, filenames in vectorized_chunks:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, vectors).flatten()
            idx = np.argmax(similarities)

            if similarities[idx] > min_similarity:
                print(similarities[idx])
                best_retrieved_texts.append(chunk[idx])
                best_filenames.append(filenames[idx])

        if best_retrieved_texts != "":
            best_response, best_filenames = generate_response(query, best_retrieved_texts, best_filenames, api_key, max_tokens=2000)
        else:
            best_response = "No suitable chunk found."
            best_filenames = 'N/A'

        if best_response:
            data = {
                'query': query,
                'query_id': query_id,
                'retrieved_text': best_retrieved_texts,
                'response': best_response,
                'filenames': best_filenames
            }
            save_to_csv(data)
            save_to_db(data)

        print(f"Filename: {best_filenames}")
        print(f"Context: {best_retrieved_texts}")
        print(f"Response: {best_response}")

# Main function
def main():
    folder_path = 'archive/'
    
    # Replace with your OpenAI API key
    api_key = OPENAI_API_KEY
    
    connection = mysql.connector.connect(user=MYSQL_USER, password=MYSQL_PASSWORD,
                              host=MYSQL_HOST,
                              port=MYSQL_PORT,
                              database=MYSQL_DATABASE)
    chatbot_with_prevectorized_chunks(api_key, min_similarity=.17)

if __name__ == "__main__":
    main()
