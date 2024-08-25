
# 🎙️ Political Debate Simulator
## 🏆 Winner of the 2024 Code the Vote Hackathon

The **Presidential Debate Simulator** is an interactive web application designed to simulate a debate between two presidential candidates. Users can ask questions, read the candidates' responses, vote on the best answers, and test their knowledge with trivia questions about US Presidents.

---

## ✨ Features

- **Interactive Debate**: Ask questions and receive responses from Candidate A and Candidate B.
- **Voting System**: Vote for the best response and see the overall results.
~~- -**Trivia Section**: Test your knowledge with trivia questions about US Presidents.~~
- **Earn Badges**: Earn badges for participation and correct answers in the trivia section.

---

## How to Use

1. **Ask a Question**: Enter your question in the input box and click "Ask".
2. **Read Responses**: Review the responses from Candidate A and Candidate B.
3. **Vote**: Vote for the response you like best by clicking the "Vote" button under the candidate's response.
4. **End Debate**: Click "End Debate" to see the results.
~~5. **Test Your Knowledge**: Answer trivia questions by selecting the correct candidate.~~

~~## Decrypting the API Key~~ NO LONGER SUPPORTED

~~To decrypt the API key on Windows, follow these steps:~~

~~1. Ensure you have OpenSSL installed on your system. If not, download and install it from [OpenSSL for Windows](https://slproweb.com/products/Win32OpenSSL.html).~~

~~2. Download the `api_key.enc` file from this repository.~~

~~3. Open a terminal (Command Prompt, PowerShell, etc.) and navigate to the directory containing `api_key.enc`.~~

~~4. Run the following command to decrypt the API key:~~
~~```sh~~
   ~~openssl enc -aes-256-cbc -d -in api_key.enc -out api_key.txt -pass pass:judge123!! -pbkdf2~~

~~For Unix (Linux/Mac):~~
~~1. Ensure you have OpenSSL installed on your system. You can install it using your package manager if it's not already installed.~~

~~For Linux:~~

~~sudo apt-get install openssl  # Debian/Ubuntu~~
~~sudo yum install openssl      # CentOS/RHEL~~
~~sudo pacman -S openssl        # Arch Linux~~

~~For Mac:~~

~~brew install openssl~~

~~2. Download the api_key.enc file from this repository.~~

~~3. Move the api_key.enc file to your desired directory (e.g., your home directory).~~

~~4. Open a terminal and navigate to the directory containing api_key.enc. For example:~~

~~cd /Desktop~~

~~5. Run the following command to decrypt the API key:~~

~~openssl enc -aes-256-cbc -d -in api_key.enc -out api_key.txt -pass pass:judge123!! -pbkdf2~~

~~6. This will create a file named api_key.txt in the same directory containing the decrypted API key.~~


## Getting Started

To run this project locally:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/presidential-debate-simulator.git
2. Navigate to directory:
   cd presidential-debate-simulator
~~3. Open script.js and paste the decrypted apiKey into line 39 and 321 and click save~~
3. Create an API via openai.com and paste the apiKey into line 39 and 321 and click save. 
   
4. Open index.html in your web browser. (Best using Chrome)

## Data Analysis and Visualizations
* View data visualizations created from mock_response_data.csv by running voting_viz.ipynb 
* Categorize raw questions into broader topics using categorize.py. This takes in questions from uncategorizedQuestions.csv and puts them through ChatGPT to classify them into categories and outputs these in a new csv. You will need to create an api key and adjust the code for this to work for you.


## 🔄 Updates

Stay informed about the latest changes and improvements to the project. Below you'll find a summary of recent updates:

> ### 08/07/2024
> 
> **Added Files:**
> 
> - `main.py`: This file includes functions from `rag_debate.py`. Use `rag_debate.py` to run inside a notebook. `main.py` requires a server.
> 
> **Purpose:**
> 
> Contains the main FastAPI application.
> 
> **Key Features:**
> 
> - Defines the FastAPI app instance.
> - Implements utility functions for text processing.
> - Includes the `QueryRequest` Pydantic model for request validation.
> - Sets up a startup event to preprocess and vectorize data.
> - Defines the `/query` endpoint to handle user queries and generate responses.
> - Uses OpenAI's GPT-4 model to generate responses based on retrieved text chunks.
---

> ### 08/11/2024
> 
> **Added Files:**
> 
> - `process_sources.py`: makes the chunks and vectorizes them and saves them into a pickle file. Also adjusted the chunking to enforce the chunks to stay within the set size
> 
> **Update:**
> 
> - `RAG Debate.py`
>   - remove the parts about processing the source files into vectors. This file is now only the user input and response generation
> 
> **Purpose:**
> 
> separates the vectorizing and chunking process to its own script so it could run apart from the game, which saves resources not reprocessing the files each game, and makes run time faster.
> Making the chunking enforce a size limit makes each chunk more focused so it helps the context retrieve relevant information 

---

> ### 08/13/2024
> 
> **Added Files:**
> 
> - `requirements.txt`
> 
> **Update:**
> 
> - `RAG Debate.py`
>   - Add - `main = 'insert model of choice'`
> 
> **Purpose:**
> 
> Ease of testing different models. As of today, `gpt-4o-mini` seems to be the best with responses.

---

> ### 08/21/2024
>
> **Renamed Files:**
>
> - `RAG Debate.py to TFID RAG Debate.py`
> - `process_sources.py to TFID_process_sources.py` Both changed to differentiate between TFID Vectorization and OpenAPI
>   
> **Added Files:**
> 
> - `OpenAPI RAG Debate.py`
> - `OpenAPI_process_sources.py`
> - `generated_client.zip`
> - `openai-api.yaml` - OpenAPI Spec File
> 
> **Purpose:**
> 
> To attempt using better vectorization via OpenAPI.
> 
> **Key Features:**
> 
> - API Client Configuration - Structured to interact with OpenAI's API, two specific models: `text-embedding-ada-002` for generating embeddings and `gpt-4o-mini` for generating chat completions. It sets up an API client using configuration classes and authorizes requests with an API key.
> - Text Preprocessing and Chunking - It includes functions for extracting, preprocessing, and chunking text data from .txt files, enabling efficient handling of large documents by breaking them into smaller, manageable pieces based on token length.
> - OpenAI Embedding Generation - The code provides a function to generate text embeddings using the `text-embedding-ada-002` model. These embeddings are crucial for comparing the similarity of different text chunks
> 
> **Requirements:**
>
> - JDK Development Kit 22.0.2 - https://www.oracle.com/java/technologies/downloads/?er=221886
> 1. In the Environment Variables window, under the System variables section, click New....
> 2. In the Variable name field, enter JAVA_HOME.
> 3. In the Variable value field, enter the path to your JDK installation directory. This will be something like: `C:\Program Files\Java\jdk-22.0.2`
> 4. Add the BIN to environment variables (path)
>
> - ONLY TO GENERATE NEW CLIENT - OpenAPI-generator -  https://repo1.maven.org/maven2/org/openapitools/openapi-generator-cli/7.7.0/openapi-generator-cli-7.7.0.jar - ONLY TO GENERATE NEW CLIENT - 
> 1. Ensure OpenAPI Spec File `openai-api.yaml` is located somewhere
> 2. Run the following - `java -jar C:\openapi-generator-cli.jar generate -i 'PATH TO "openai-api.yaml"' -g python -o ./generated-client`
> 3. You can install the generated client into your Python environment by navigating to the generated-client directory and running: `pip install .`
> 4. By default, the generated-client folder is created in the current working directory from which you run the OpenAPI Generator command.
>   
> ### 08/25/2024
> 
> **Update:**
> 
> - `TFID RAG Debate.py`
>   - Add RAGAS faithfulness and answer_relevancy scores for each query response
>   - Save the responses and scores to mysql
> 
> **Purpose:**
> 
>  The RAGAS scores are to keep a pulse on how good our chat bot answers are. These can be analyzed to help us improve the application.
> 

## 📜 License

This project is licensed under the **Testing Purposes License**. For more details, please see the [LICENSE](./LICENSE) file.


## 📚 Text Data Sources

Here are the key documents and speeches used as text data sources for our project. Each link provides access to publicly available content for analysis.

### 🗣️ Speeches and Public Statements

> **Donald Trump**  
> [📄 View Document](https://docs.google.com/document/d/12vgqTrVF0JSSBvXW6xwxdUVT9w36WXVVYiTvjRKfRzw/edit)  
> *A collection of speeches and public statements made by Donald Trump.*

> **Kamala Harris**  
> [📄 View Document](https://docs.google.com/document/d/1-m0UCzJ7CY_NwdJid91wIa0JwDxwfkBYnpSS6FDiHAg/edit)  
> *Key speeches and public addresses by Vice President Kamala Harris.*

### 🏛️ Political Figures

> **Governors' Statements**  
> [📄 View Document](https://docs.google.com/document/d/16OO5ZqDZtAyE6GW79tesC7MaJ5z5jiVP2HwZ7Fu4VyI/edit)  
> *Statements and announcements from various U.S. governors.*

> **Steve Garvey**  
> [📄 View Document](https://docs.google.com/document/d/16RuK5aP-nP5hRO_TT7V9IV14zJStBbt_6cqh84_C--I/edit)  
> *Public statements and interviews with Steve Garvey.*

> **Adam Schiff**  
> [📄 View Document](https://docs.google.com/document/d/16TBYK2v3isS1D8b_RjTntDKzZKMZ8d2TmDlX_oFQs7M/edit)  
> *Speeches and public comments from Representative Adam Schiff.*

> **Ted Cruz**  
> [📄 View Document](https://docs.google.com/document/d/16gHSkshQ2EwIAF-XKYd5AlSoN5Sor-5aOQucSlW2Msw/edit)  
> *Public addresses and policy statements from Senator Ted Cruz.*

> **Colin Allred**  
> [📄 View Document](https://docs.google.com/document/d/16cnauFJRgS2Wpp5RnTaieCUAr3aki063SZFX0cAuMVY/edit)  
> *Key speeches and positions articulated by Representative Colin Allred.*

