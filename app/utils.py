import os
import pickle
import re
import csv
import json
import numpy as np
from scipy.spatial.distance import cosine
import tiktoken  # OpenAI's tokenizer
import pandas as pd
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from datasets import Dataset 
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import openai
import sqlalchemy
from datetime import datetime
from openai import OpenAI


client = openai.OpenAI()

# Correctly load the .env file
#dotenv_path = r"C:\Users\Patra\OneDrive\Documents\GitHub\debate_bot\app\.env"
#load_dotenv(dotenv_path=dotenv_path)

# Get API key from environment and debug print to check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print("Loaded API Key:", api_key)  # Debugging print statement

if api_key:
    # Initialize OpenAI with the loaded API key
    openai.api_key = api_key
else:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")
    
# Initialize OpenAI API key
#openai.api_key = os.getenv('OPENAI_API_KEY')

MYSQL_DATABASE = 'debatebot_dev'

# Database Configuration
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
#MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

# Model to use for chat completions
model = 'gpt-4o-mini'

# Debugging: Print environment variable values
print(f"MySQL User: {MYSQL_USER}")
print(f"MySQL Host: {MYSQL_HOST}")
print(f"MySQL Database: {MYSQL_DATABASE}")

# Function to create a database connection using MySQL Connector
def get_database_connection():
    try:
        print(f"Connecting to database: {MYSQL_DATABASE}")  # Debugging statement
        connection = mysql.connector.connect(
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            database=MYSQL_DATABASE
        )
        return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        raise e


# Function to create a database connection using SQLAlchemy
def get_database_engine():
    try:
        engine = create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')
        return engine
    except Exception as e:
        print(f"Error creating SQLAlchemy engine: {e}")
        raise e

def insert_into_database(sql_string, vals):
    print(f"Executing SQL: {sql_string} with values {vals}")  # Debugging
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # Ensure correct number of parameters for the SQL insert
        if 'INSERT INTO Query' in sql_string:
            # Make sure vals has the exact number of parameters that matches the SQL
            if len(vals) == 2:
                vals = (vals[0], vals[1], datetime.now())  # Add current timestamp

        cursor.execute(sql_string, vals)
        connection.commit()
        print(cursor.rowcount, "record inserted.")
        
    except Error as e:
        print(f"Error executing insert: {e}")
        raise e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def select_from_database(sql_string):
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        cursor.execute(sql_string)
        records = cursor.fetchall()
        return records
    except Error as e:
        print(f"Error executing select: {e}")
        raise e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to estimate the token length of text using OpenAI's tokenizer
def estimate_tokens(text, model_name=model):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to truncate text to fit a specified number of tokens
def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_texts, max_tokens=4096):
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
        response = openai.chat.completions.create(
            model=model,
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

# Function to save data to a CSV file
def save_to_csv(data, filename='chatbot_data.tsv'):
    file_exists = os.path.isfile(filename)
    # Open the file with UTF-8 encoding
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filenames']])

# Function to save data to the database
def save_to_db(data):
    contexts = json.dumps(data['retrieved_text'])
    filenames = json.dumps(data['filenames'])
    vals = (data['query_id'], 1, data['response'], contexts, filenames, 0, float(data['answer_relevancy']), float(data['faithfulness'])) #TODO fill in candidate and vote
    insert_into_database(f"INSERT INTO Response (queryId, candidateId, response, contexts, filenames, userVoted, answerRelevancyScore, faithfulnessScore) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", vals)

# Function to generate OpenAI embeddings for a given text
def get_openai_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# Function to extract URL from the top of a text file
def extract_url_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the first line where the URL is expected
            url = file.readline().strip()
            # Validate if it is a URL
            if re.match(r'(https?://\S+)', url):
                return url
            else:
                print(f"No valid URL found at the top of {file_path}.")
                return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading URL from {file_path}: {e}")
        return None

# Function to find the best matching texts based on cosine similarity
def find_best_texts(query_embedding, pkl_filenames, txt_folder_path, n):
    best_retrieved_texts = []
    best_filenames = []
    best_similarities = []
    best_urls = []  # List to store URLs

    # Dictionary to cache URLs extracted from .txt files
    url_cache = {}

    # Process each .pkl file and retrieve texts
    for pkl_filename in pkl_filenames:
        with open(pkl_filename, 'rb') as file:
            vectorized_chunks = pickle.load(file)

            # Process each chunk in the .pkl file
            for embedding, chunk, chunk_filenames, chunk_urls in vectorized_chunks:
                # Extract the corresponding .txt filename
                txt_filename = chunk_filenames[0]  # Assuming chunk_filenames contains the original .txt filename
                
                # Compute similarity and store the best results
                similarity_score = cosine(query_embedding, embedding)
                if similarity_score > 0:
                    best_similarities.append(similarity_score)
                    best_retrieved_texts.append(chunk[0])
                    best_filenames.append(chunk_filenames[0])
                    best_urls.append(chunk_urls[0])  # Use the URL from the .pkl file

    # Combine results into a DataFrame and sort by similarity
    text_similarities = pd.DataFrame(
        {
            'texts': best_retrieved_texts,
            'filenames': best_filenames,
            'similarities': best_similarities,
            'urls': best_urls  # Include URLs in the DataFrame
        }
    )

    result = text_similarities.sort_values('similarities', ascending=True)

    # Print the top results for debugging
    print(result.head())

    # Return the top 'n' results
    return result.head(n)

# Function to compute scoring metrics
def get_scoring_metrics(query, response, contexts):
    data = {
        'question': [query],
        'answer': [response],
        'contexts' : [contexts]
    }
    dataset = Dataset.from_dict(data)
    score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    score.to_pandas()
    return score

