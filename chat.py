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

#
class AgentType(enum.Enum):
    PANDAS_AGENT = "pandas_agent"

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,

}





def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    prompt = (
            """
                For the following query, if it requires drawing a table, reply as follows:
                {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
    
                If the query requires creating a bar chart, reply as follows:
                {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
    
                If the query requires creating a line chart, reply as follows:
                {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
    
                There can only be two types of chart, "bar" and "line".
    
                If it is just asking a question that requires neither, reply as follows:
                {"answer": "answer"}
                Example:
                {"answer": "The title with the highest rating is 'Gilead'"}
    
                If you do not know the answer, reply as follows:
                {"answer": "I do not know."}
    
                Return all output as a string.
    
                All strings in "columns" list and data list, should be in double quotes,
    
                For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}
    
                Lets think step by step.
    
                Below is the query.
                Query: 
                """
            + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return str(response)


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
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def run_query(agent, query_):
    #if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
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

def clear_submit():
    st.session_state["submit"] = False


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


st.set_page_config(page_title="Chat with Analytical Data", page_icon="🦜")
st.title("🦜 Chat with Analytical Data")

data_directory = "csv"  # Replace with the path to the directory containing your files

file_list = os.listdir(data_directory)
selected_file = st.selectbox("Select a Data file", file_list, help="Various File formats are Support")

if selected_file:
    file_path = os.path.join(data_directory, selected_file)
    df = load_data(file_path)
    st.write(df.head())
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.text(msg["role"] + ": " + msg["content"])  # Display role and content

if prompt := st.text_input("User input", key="user_input", placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = OpenAI(temperature=0,openai_api_key=openai_api_key)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        #return_intermediate_steps=True,
        verbose=True,
        #agent_type=AgentType.PANDAS_AGENT,  # Use PANDAS_AGENT instead of OPENAI_COMPLETION
)
    #agent = create_pandas_dataframe_agent(
        #ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
       # df,
        #verbose=True)
    with st.spinner("Thinking..."):
        #st.write("Intermediate Step 1: Preparing DataFrame...")
        #st.write(df.head())
        #response, intermediate_steps,thought, action, action_input, observation= run_query(pandas_df_agent, st.session_state.messages[-1:])
        #answer = pandas_df_agent.run(st.session_state.messages)  # Pass the latest message to the agent
        #st.write(answer)
        #st.sidebar.write("Thinking process:")
        #st.sidebar.write(thought, action, action_input, observation)

        response = query_agent(pandas_df_agent,prompt)
        st.write(response)
        # Decode the response.
        decoded_response = decode_response(response)

        # Write the response to the Streamlit app.
        write_response(decoded_response)
        #st.write("Thinking process:" ,intermediate_steps)
        #st.write('====')



