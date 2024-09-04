import openai
import os
import pickle
import csv
import json
import numpy as np
from scipy.spatial.distance import cosine
import tiktoken  # OpenAI's tokenizer
import pandas as pd
from dotenv import load_dotenv
import mysql.connector
from datetime import datetime
from datasets import Dataset 
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

# Model to use for chat completions
model = 'gpt-4o-mini'

#load envrionment variables
load_dotenv()
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

def get_scoring_metrics(query, response, contexts):
    """
    This is an example of Google style.

    Args:
        param1: This is the first param.
        param2: This is a second param.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """
    data = {
        'question': [query],
        'answer': [response],
        'contexts' : [contexts]
        }
    dataset = Dataset.from_dict(data)
    score = evaluate(dataset,metrics=[faithfulness, answer_relevancy])
    score.to_pandas()
    return score

def get_database_connection():
    return mysql.connector.connect(
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        database=MYSQL_DATABASE
    )

# Function to create a database connection and run a query
def insert_into_database(sql_string, vals):
    print(sql_string)
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(sql_string, vals)
    connection.commit()
    print(cursor.rowcount, "record inserted.")
    connection.close()

def select_from_database(sql_string):
    connection = get_database_connection()
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
def estimate_tokens(text):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

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
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filenames']])

def save_to_db(data):
    contexts = json.dumps(data['retrieved_text'])
    filenames = json.dumps(data['filenames'])
    vals = (data['query_id'], 1, data['response'], contexts, filenames, 0, float(data['answer_relevancy']), float(data['faithfulness'])) #TODO fill in candidate and vote
    insert_into_database(f"INSERT INTO Response (queryId, candidateId, response, contexts, filenames, userVoted, answerRelevancyScore, faithfulnessScore) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", vals)

def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

   
def get_openai_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        return openai.embeddings.create(input = [text], model=model).data[0].embedding
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
        
# Get the texts associated with the top n best similarity. Takes in a query embedding. Returns a dataframe
def find_best_texts(query_embedding, filenames, n):
    best_retrieved_texts = []
    best_filenames = []
    best_similarities = []

    # Process each file and retrieve texts
    for filename in filenames:
        with open(filename, 'rb') as file:
            vectorized_chunks = pickle.load(file)

            for embedding, chunk, chunk_filenames in vectorized_chunks:  # Use a different name for clarity
                similarity_score = cosine(query_embedding, embedding)
                # Only consider positive similarity scores
                if similarity_score > 0:
                    best_similarities.append(similarity_score)
                    best_retrieved_texts.append(chunk[0])
                    best_filenames.append(chunk_filenames[0])  # Use correct index if needed

    # Combine results into a DataFrame and sort by similarity
    text_similarities = pd.DataFrame(
        {
            'texts': best_retrieved_texts,
            'filenames': best_filenames,
            'similarities': best_similarities
        }
    )

    result = text_similarities.sort_values('similarities', ascending=True)

    # Print the top results for debugging
    print(result.head())

    # Return the top 'n' results
    return result.head(n)


# Chatbot function
def chatbot_with_prevectorized_chunks():
    session_id = 0  # TODO replace this with real session

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        vals = (session_id, query, datetime.now())
        insert_into_database(f"INSERT INTO Query (sessionId, query, timestamp) VALUES (%s, %s, %s)", vals)
        
        query_id, query = get_last_query_from_db()

        # Get the embedding for the user's query
        query_embedding = get_openai_embedding(query)

        if query_embedding is None:
            print("Failed to get query embedding. Try again.")
            continue
        
        # Retrieve texts for Reichert
        best_texts_df_reichert = find_best_texts(query_embedding, ['vectorized_chunks_reichert.pkl'], 4)  # Ensure correct filename
        best_retrieved_texts_reichert = best_texts_df_reichert["texts"].tolist()
        best_filenames_reichert = best_texts_df_reichert["filenames"].tolist()

        # Generate a response for Reichert
        if best_retrieved_texts_reichert:
            best_response_reichert = generate_response(query, best_retrieved_texts_reichert)
        else:
            best_response_reichert = "No suitable chunk found for Reichert."
            best_filenames_reichert = 'N/A'

        # Handle response and save data for Reichert
        if best_response_reichert:
            scores_reichert = get_scoring_metrics(query, best_response_reichert, best_retrieved_texts_reichert)
            data_reichert = {
                'query': query,
                'query_id': query_id,
                'retrieved_text': best_retrieved_texts_reichert,
                'response': best_response_reichert,
                'filenames': best_filenames_reichert,
                'faithfulness': scores_reichert['faithfulness'],
                'answer_relevancy': scores_reichert['answer_relevancy']
            }
            save_to_csv(data_reichert)
            save_to_db(data_reichert)

        # Print the response for Reichert
        print(f"Response for Reichert: {best_response_reichert}")

        # Retrieve texts for Ferguson
        best_texts_df_ferguson = find_best_texts(query_embedding, ['vectorized_chunks_ferguson.pkl'], 4)  # Ensure correct filename
        best_retrieved_texts_ferguson = best_texts_df_ferguson["texts"].tolist()
        best_filenames_ferguson = best_texts_df_ferguson["filenames"].tolist()

        # Generate a response for Ferguson
        if best_retrieved_texts_ferguson:
            best_response_ferguson = generate_response(query, best_retrieved_texts_ferguson)
        else:
            best_response_ferguson = "No suitable chunk found for Ferguson."
            best_filenames_ferguson = 'N/A'

        # Handle response and save data for Ferguson
        if best_response_ferguson:
            scores_ferguson = get_scoring_metrics(query, best_response_ferguson, best_retrieved_texts_ferguson)
            data_ferguson = {
                'query': query,
                'query_id': query_id,
                'retrieved_text': best_retrieved_texts_ferguson,
                'response': best_response_ferguson,
                'filenames': best_filenames_ferguson,
                'faithfulness': scores_ferguson['faithfulness'],
                'answer_relevancy': scores_ferguson['answer_relevancy']
            }
            save_to_csv(data_ferguson)
            save_to_db(data_ferguson)

        # Print the response for Ferguson
        print(f"Response for Ferguson: {best_response_ferguson}")


# Main function
def main():

    chatbot_with_prevectorized_chunks()

if __name__ == "__main__":
    main()