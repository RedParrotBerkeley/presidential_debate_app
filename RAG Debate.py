import openai
from docx import Document
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a docx file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return full_text  # Return a list of paragraphs

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
    return paragraphs[idx]

# Function to generate a response using the ChatGPT model
def generate_response(query, vectorizer, vectors, paragraphs, api_key):
    openai.api_key = api_key
    retrieved_text = search(query, vectorizer, vectors, paragraphs)
    prompt = f"Context: {retrieved_text}\nQuestion: {query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are providing responses based on the content of the provided document about Donald Trump, never refer to the document but act like you are answering as Donald Trump. Stick to the facts and avoid mannerisms, never refer to yourself as Donald Trump. Keep the answer short, clear, and concise."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    # Extract the relevant portion of the generated text
    generated_text = response.choices[0]['message']['content'].strip()
    return generated_text

def main():
    file_path = 'word.docx'
    raw_paragraphs = extract_text_from_docx(file_path)
    processed_paragraphs = preprocess_paragraphs(raw_paragraphs)
    
    vectorizer, vectors = vectorize_paragraphs(processed_paragraphs)
    
    # Replace with OpenAI API key
    api_key = 'Paste from slack do NOT commit changes with API key'

    def chatbot():
        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit']:
                break
            response = generate_response(query, vectorizer, vectors, processed_paragraphs, api_key)
            print(f"Candidate A: {response}")

    chatbot()

if __name__ == "__main__":
    main()