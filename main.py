import openai
from langchain.llms import OpenAI
import pandas as pd
import os
from csv_chatbot import create_csv_chatbot
import json
import matplotlib.pyplot as plt
import numpy as np

ALLOW_PLOTTING=False
prefix = """You are working with a pandas dataframe in Python. The name of the dataframe is `df1`. 
            You have access to models you have previously created. 
            The name of model files are: {model_files.pkl}.
            Load these model files using pickle as model instead of creating a new model based on df1, preprocess data using Label Encoding,fill N/A, make prediction  ,  output r2 value ,reply the model is loaded.
            If  you found no such file, create a new regression model, preprocess data using Label Encoding,fill N/A,define the input, make prediction based on the loaded dataset,then save it to the current directory using pickle with name as model_files,output r2_score,reply create a new model.
            This saved model file should uniquely identify the model and associated task for your later use.
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [[25, 24, 10, ...],[25, 24, 10, ...]]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [[25, 24, 10, ...],[25, 24, 10, ...]]}}

            If it is just asking a question that doesn't request crate charts or tables, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
# Function to decode response from agent
def decode_response(response: str) -> dict:
    if not ALLOW_PLOTTING:
        return response
    if ("\"table\"" in response or "\"bar"  in response or "\"line\""  in response):
        return json.loads(response)
    else:
        return response
    
def write_response(response_dict: dict):
    if not ALLOW_PLOTTING:
        return response_dict
    if "\"table\"" in response:
        data = response_dict["table"]
        table_data= []
        table_data.append(list(data['columns']))
        for row in data['data']:
            table_data.append(list(row))
        plt.table(cellText=table_data, loc='center')
        plt.axis('off')
        plt.show()
        return f"Here I displayed your table"
  
    elif "\"bar\"" in response:
        data = response_dict["bar"]
        plt.hist(np.squeeze(np.array(data['data'])))
        plt.ylabel("Count")
        plt.xlabel(data['columns'][0])
        plt.show()
        return "Here I displayed your bar chart!"
    
    elif "\"line\"" in response:
        data = response_dict["line"]
        xs = [l[0] for l in data['data']]
        ys = [l[1] for l in data['data']]
        plt.plot(list(xs),list(ys),'*-')
        plt.show()
        return "Here I displayed line chart!"
    else:
        return response_dict
     
def read_from_txt(path):
    with open(path) as f:
        text = f.readlines()
    return text[0]

# load api key
os.environ['OPENAI_API_KEY'] = read_from_txt("openai_key.txt")

welcome_text=read_from_txt("welcome.txt")

user_input = input(welcome_text+"\n")
csv_names = [pd.read_csv(df_path) for df_path in user_input.replace(" ",'').split(",")]

agent = create_csv_chatbot(OpenAI(temperature=0), csv_names, verbose=True)

# ask the question ex: What is the average fare?
while((query:=input("Enter your query\n"))!="exit"):
    query = prefix + query
    response = agent.run(query) 
    decoded_response = decode_response(response.__str__())
    write_response(decoded_response)
