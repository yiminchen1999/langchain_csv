import streamlit as st
from functions import *
import platform
from streamlit_image_select import image_select
import subprocess
import streamlit as st
import os
from streamlit_chat import message
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import plotly.figure_factory as ff

import numpy as np
import matplotlib.pyplot as plt
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,

}
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,

}
def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save it and show these charts or plots in your response.'
    query += ' . ' + q_s
    return query

data_directory = "csv"  # Replace with the path to the directory containing your files
file_list = os.listdir(data_directory)


def run_query(pdagent, query_):
    #if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
    if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
        save_chart(query_)
    output = pdagent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    return response, thought, action, action_input, observation

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

def show_data(tabs, df_arr):
    for i, df_ in enumerate(df_arr):
        print(i, len(df_))
        with tabs[i]:
            st.dataframe(df_)

def get_text(n):
    input_text = st.text_input('How can I help?', '', key="input{}".format(n))
    return input_text

def main():

    st.title("Chat with your CSV")

    selected_file = st.selectbox("Select a Data file", file_list, help="Various File formats are Support")
    if selected_file:
        file_path = os.path.join(data_directory, selected_file)
        df = load_data(file_path)
        x = 0
        user_input = get_text(x)
        if st.button("Submit Query", type="primary"):
            x += 1
        # Create an agent from the CSV file.
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            pdagent = create_pandas_dataframe_agent(llm, df, verbose=True,return_intermediate_steps=True)
            with st.spinner("Thinking..."):
                print(user_input, len(user_input))
                response, thought, action, action_input, observation = run_query(pdagent, user_input)
                for i in range(0, len(thought)):
                    #if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' or 'draw' or 'drew' in user_input:
                        #group_labels = df.columns.tolist()
                    #st.pyplot(df.plot())
                    #st.write(df.describe())
                    st.line_chart(df)
                    st.bar_chart(df)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)

            # Display the generated response messages
                for i in range(len(st.session_state['generated']) - 1, -1, -1):

                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                # Display the plots
                for i in range(0, len(thought)):
                    st.sidebar.write(thought[i])
                    st.sidebar.write(action[i])
                    st.sidebar.write(action_input[i])
                    st.sidebar.write(observation[i])
                    st.sidebar.write('====')
                    if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' or 'draw' or 'drew' in user_input:
                        st.write(action_input[i])


if __name__ == "__main__":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'tabs' not in st.session_state:
        st.session_state['tabs'] = []

    main()





