import os
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# Prompt the user for their OpenAI API key
api_key = input("Please enter your OpenAI API key: ")

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

# Optionally, check that the environment variable was set correctly
print("OPENAI_API_KEY has been set!")

# load documents from a directory into a documents object
loader = DirectoryLoader("sources/")
documents = loader.load()

for document in documents:
    document.metadata['filename'] = document.metadata['source'].split('/')[-1]
    print(document.metadata['filename'])


# make generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

# save testset to a csv
testset.to_pandas().to_csv('testsets/test_dataset.csv', index=False)
