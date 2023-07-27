import streamlit as st
import pandas as pd
import json
from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import enum
from langchain.llms import OpenAI
import seaborn as sns
import json
from agent import query_agent

from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,

}
def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save them localy and include the save file names in your response.'
    query += ' . ' + q_s
    return query


def load_data(file_path):
    try:
        ext = os.path.splitext(file_path)[1][1:].lower()
    except:
        ext = file_path.split(".")[-1]
    if ext in file_formats:
        df = file_formats[ext](file_path)
        return df
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


def run_query(agent, query_):
    #if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
    if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    return response, intermediate_steps,thought, action, action_input, observation

def decode_intermediate_steps(steps):
    log, thought_, action_, action_input_, observation_ = [], [], [], [], []
    text = ''
    #把thinking process extract出来
    for step in steps:
        thought_.append('[{}]'.format(step[0][2].split('Action:')[0]))
        action_.append('[Action:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[0]))
        action_input_.append(
            '[Action Input:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[1]))
        observation_.append('[Observation:] {}'.format(step[1]))
        log.append(step[0][2])
        text = step[0][2] + ' Observation: {}'.format(step[1])
    return thought_, action_, action_input_, observation_, text

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)


def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        df = response_dict["line"]
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        st.table(df)
    if "plot" in response_dict:
        st.plot(df)


data_directory = "csv"  # Replace with the path to the directory containing your files

file_list = os.listdir(data_directory)
st.title("Chat with your CSV")
selected_file = st.selectbox("Select a Data file", file_list, help="Various File formats are Support")


if selected_file:
    file_path = os.path.join(data_directory, selected_file)
    df = load_data(file_path)
    st.write(df.head())

query = st.text_area("Insert your query")

if st.button("Submit Query", type="primary"):

    # Create an agent from the CSV file.
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True,return_intermediate_steps=True)
    with st.spinner("Thinking..."):
        response = run_query(agent, query)
        st.write(response)
    # Decode the response.
        decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
        write_response(decoded_response)
