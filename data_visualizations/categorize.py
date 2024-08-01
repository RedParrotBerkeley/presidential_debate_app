import os
import csv
from openai import OpenAI

"""
this script opens a file inFilename, and takes questions from the first column
the questions and sent to chatgpt to categorize the questions.
    Note: need active API Key with tokens
the categories are appended to end of each line of the question in a new output .csv
"""

chatOn = False #set to True if actually communicating with ChatGPT
inFilename = "uncategorizedQuestions.csv"

#open chat client
client = OpenAI(
    api_key="insertyourapikeyhere",
    organization='org-D5PVVCgcRCZmBqsXopnpc4Y4',
#   project='proj_HC153goxmLyr1joZKTO0Qeik'
)

#read open file
fileData = []
with open(inFilename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        #read file data into a list
        fileData.append(row)

#create header for new file
newHeader = fileData[0] + ["Category"]
newFileData = [newHeader]

prompts=[{"role": "system", "content": "You are a political scientist at a national newspaper"}]
prompts.append({"role": "system", "content": "categorize next statements in 1 word as a White House correspondant"})

#for each line, get question, send to chatgpt, and retrieve category
for line in fileData[1:]:
    question = line[0]

    if chatOn:
        #add question to prompt
        prompts.append({"role": "citizen", "content": question})

        answers = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=20,
            messages=prompts,
            top_p=0.1
        )

        #remove question from prompt
        prompts.pop()

        response = answers['choices'][0]['message']['content']
        category = response
    else:
        category = "Test"
    newLine = line + [category]
    newFileData.append(newLine)

#append category to line, and output to new file
outFilename = inFilename.replace(".csv","Out.csv")
with open(outFilename, 'w',newline='') as f:
    csv_writer = csv.writer(f,delimiter=',')
    csv_writer.writerows(newFileData)