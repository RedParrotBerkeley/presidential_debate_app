import os
import csv
from openai import OpenAI

"""
this script opens a file inFilename, and takes questions from the first column
the questions and sent to chatgpt to categorize the questions.
    Note: need active API Key with tokens
the categories are appended to end of each line of the question in a new output .csv
"""

chatOn = True #set to True if actually communicating with ChatGPT
inFilename = "uncategorizedQuestions_small.csv"

#open chat client
client = OpenAI(
    api_key="your-api-key-here"
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

prompts=[{"role": "system", "content": 
    """
    You are a political scientist at a national newspaper. 
    Categorize the next statements to their most relevant political key issue.
    If the statement is not similar to any key issue, then categorize it as Other.
    The key issues are:
    Economy,
    Healthcare,
    Education,
    Immigration,
    Environment,
    National Security,
    Criminal Justice,
    Social Justice and Civil Rights,
    Tax Policy,
    Gun Control,
    Infrastructure,
    Public Safety,
    Foreign Policy,
    Housing,
    Social Welfare Programs,
    Drug Policy,
    Veterans Affairs,
    Technology and Privacy,
    Election Integrity,
    Reproductive Rights,
    Gender,
    Religious Freedom
    
    """
    }]

#for each line, get question, send to chatgpt, and retrieve category
for line in fileData[1:]:
    question = line[0]

    if chatOn:
        #add question to prompt
        prompts.append({"role": "user", "content": question})

        answers = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=20,
            messages=prompts,
            top_p=0.1
        )

        #remove question from prompt
        prompts.pop()

        category = answers.choices[0].message.content
        print(category)
    else:
        category = "Test"
    newLine = line + [category]
    newFileData.append(newLine)

#append category to line, and output to new file
outFilename = inFilename.replace(".csv","Out.csv")
with open(outFilename, 'w',newline='') as f:
    csv_writer = csv.writer(f,delimiter=',')
    csv_writer.writerows(newFileData)