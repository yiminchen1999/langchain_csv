
from typing import Any, List, Optional
from langchain.agents.agent import AgentExecutor
from langchain.agents import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory


from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import glob
import json
from datetime import datetime


def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save them localy and include the save file names in your response.'
    query += ' . ' + q_s
    return query


def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df_arr, df_arr_names = load_dataframe()

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df_arr, return_intermediate_steps=True,
                                          save_charts=True, verbose=True)
    return agent, df_arr, df_arr_names


def load_dataframe():
    selected_df = []

    all_files_csv = glob.glob("*.csv")
    all_files_xlsx = glob.glob("*.xlsx")
    all_files_xls = glob.glob("*.xls")
    for filename in all_files_csv:
        df = pd.read_csv(filename)
        selected_df.append(df)
    for filename in all_files_xlsx:
        df = pd.read_excel(filename)
        selected_df.append(df)
    for filename in all_files_xls:
        df = pd.read_excel(filename)
        selected_df.append(df)
    selected_df_names = all_files_csv + all_files_xlsx + all_files_xls
    return selected_df, selected_df_names


def run_query(agent, query_):
    if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    return response, thought, action, action_input, observation


def decode_intermediate_steps(steps):
    log, thought_, action_, action_input_, observation_ = [], [], [], [], []
    text = ''
    #把thinking process extract出来
    for step in steps:
        thought_.append(':green[{}]'.format(step[0][2].split('Action:')[0]))
        action_.append(':green[Action:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[0]))
        action_input_.append(
            ':green[Action Input:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[1]))
        observation_.append(':green[Observation:] {}'.format(step[1]))
        log.append(step[0][2])
        text = step[0][2] + ' Observation: {}'.format(step[1])
    return thought_, action_, action_input_, observation_, text
