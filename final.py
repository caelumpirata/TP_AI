import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import pygwalker
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, FewShotPromptTemplate, \
    PromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from streamlit_chat import message
import tempfile
import psycopg2
from langchain.agents.agent import AgentExecutor
from mitosheet.streamlit.v1 import spreadsheet
from langchain.agents import AgentType
from langchain.memory import ConversationBufferWindowMemory
import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import re
import random
from streamlit.components.v1 import html
from langchain.agents import create_openai_tools_agent
import matplotlib.pyplot as plt


# Function to establish a connection to the PostgreSQL database
def create_database_connection(host, user, password, database, port):
    try:
        conn = psycopg2.connect(
            host=host, user=user, password=password, database=database, port=port,
        )
        return conn
    except Exception as e:
        st.error(e)
        st.stop()


memory = ConversationBufferWindowMemory(k=7)

pg_uri = "postgresql+psycopg2://master:0r5VB[TL?>A/8,}<vkpmEwS)@65.20.77.132:32432/ems_ai"
db = SQLDatabase.from_uri(pg_uri)


# Function to save number to a text file
def save_number_to_file(number):
    with open("numbers.txt", "a") as file:
        file.write("\n" + str(number))


# Function to read numbers from file and return as a list
def read_numbers_from_file():
    with open("numbers.txt", "r") as file:
        numbers = [line.strip() for line in file.readlines()]
    return numbers


table_connections = {}
tables = []

st.session_state['x'] = 1
st.session_state.chats = ["Chat 01"]

with open(r'config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

output = None

authenticator.login()

# try:
#     email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
#     if email_of_registered_user:
#         st.success('User registered successfully')

#         with open(r'config.yaml', 'w') as file:
#             yaml.dump(config, file, default_flow_style=False)

# except Exception as e:
#     st.error(e)


if st.session_state["authentication_status"]:

    full_name = st.session_state["name"]
    name_parts = full_name.split()
    initials = ''.join([part[0].upper() for part in name_parts])

    authenticator.logout(initials)

    st.markdown("""
    <style>

    header.st-emotion-cache-18ni7ap.ezrtsby2{
    display:none;
    }

    div.st-emotion-cache-10zg0a4.eczjsme1{
    display:none;
    }

    div.row-widget.stButton[data-testid="stButton"]{

    right: -40.5rem;
    width: 80px;
    top: 2%;
    position: fixed;
    z-index: 10000;
    }

    button.st-emotion-cache-7ym5gk.ef3psqc12[data-testid="baseButton-secondary"]{
    border-radius: 50%;
    width: 40px;

    }

    </style>
    """, unsafe_allow_html=True)

    index1 = 2
    if f"history2{index1}" not in st.session_state:
        st.session_state[f"history2{index1}"] = []

    if f"past2{index1}" and f"generated2{index1}" not in st.session_state:
        st.title(f"Welcome {full_name}")
        st.session_state[f"generated2{index1}"] = []
        st.session_state[f"past2{index1}"] = []

    # container for the chat history
    response_container = st.container()

    user_input = st.chat_input("Chat here!")
    container = st.container()
    with st.container():

        # -----------------------------------NLP Graphs-----------------------------------------------------

        if user_input:
            with st.spinner("Searching for answer..."):
                examples = [
                    {"input": "what is the maximum y phase voltage?", "answer": "3"},
                    {
                        "input": "what is the average current of y phase?",
                        "answer": "3",
                    },
                    {
                        "input": "delete values of voltage when its null",
                        "answer": "3",
                    },
                    {
                        "input": "what in the total consumption of kwh in monsoon?",
                        "answer": "3",
                    },
                    {
                        "input": "How often does the voltage of ry phase exceed the upper limit specified by regulations?",
                        "answer": "3",
                    },
                    {
                        "input": "What is the current behavior during monsoon season ?",
                        "answer": "1",
                    },
                    {
                        "input": "show me chart for voltage of r phase vs time",
                        "answer": "2",
                    },
                    {
                        "input": "what types of charts can you generate?",
                        "answer": "1",
                    },
                    {
                        "input": "how line charts and bar charts are different from each other?",
                        "answer": "1",
                    },
                    {
                        "input": "show me consumption of energy of march",
                        "answer": "3",
                    },
                    {
                        "input": "what will be good for my data line chart or bar chart?",
                        "answer": "1",
                    },
                    {
                        "input": "what is the upper limit of voltage?",
                        "answer": "1",
                    },
                    {
                        "input": "can i generate pie charts based on my data?",
                        "answer": "1",
                    },
                    {
                        "input": "show me energy consumption of 04 meter of glide location in the jan 2023",
                        "answer": "3",
                    },
                    {
                        "input": "plot bar chart for kwh vs time for meter id 0003",
                        "answer": "2",
                    },
                    {
                        "input": "explain the data to me",
                        "answer": "1",
                    },
                    {
                        "input": "how does voltage vary from all phases in peak hours?",
                        "answer": "1",
                    },
                    {
                        "input": "how does kwh changing in peak hours?",
                        "answer": "1",
                    },
                ]

                system_prefix = """You are a project manager who manage that tasks.
                        you have to identify what user actually wants.
                        you have energy meter data, and user will ask anything about it.
                        If you understand that user is asking for any type of summmerization or query related to data analysis, then just return "1" as the answer.
                        If you understand that user is asking for visualization of data or asking for generating graphs then just return "2" as the answer.
                        If you understand that user is asking for sql data or anything related to generating sql queries and giving answer, then just return "3" as the answer.

                        NOTE : 
                        You are not allowed to use a single string or any other words in your output.
                        It should always be 1,2 or 3.
                        Never generate any output which contains any other word or digit besides 1,2 and 3.

                        """

                few_shot_prompt = FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=PromptTemplate.from_template(
                        "User input: {input}\n answer: {answer}"
                    ),
                    input_variables=["input", "dialect", "top_k"],
                    prefix=system_prefix,
                    suffix="",
                )

                full_prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessagePromptTemplate(prompt=few_shot_prompt),
                        ("human", "{input}"),
                    ]
                )

                llm = ChatGroq(temperature=0.1,
                               groq_api_key="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",
                               model_name="llama3-8b-8192")

                chain = full_prompt | llm

                res = chain.invoke({"input": user_input})
                print(res.content)

                if res.content == "2" or res.content == "Answer: 2":
                    try:
                        conn = psycopg2.connect(
                            host="65.20.77.132", user="master", password="0r5VB[TL?>A/8,}<vkpmEwS)", database="ems_ai",
                            port="32432",
                        )
                        llm = ChatGroq(temperature=0.1,
                                       groq_api_key="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",
                                       model_name="llama3-8b-8192")

                        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                        context = toolkit.get_context()
                        tools = toolkit.get_tools()

                        examples1 = [
                            {
                                "input": "plot chart of current behavior during monsoon season ?",
                                "query": 'SELECT "avg", "time" FROM current_total WHERE EXTRACT(month from time) in (6,7,8,9); ',
                            },
                            {
                                "input": "plot graph for y phase voltage vs time in february?",
                                "query": 'SELECT "avg", "time" FROM voltage_y WHERE EXTRACT(month from time) in (2);',
                            },
                            {
                                "input": "show me graph for average current across all phases vs time?",
                                "query": 'SELECT "avg","time" FROM current_total LIMIT 100;',
                            },
                            {
                                "input": "show me chart for consumption of kwh in monsoon?",
                                "query": 'SELECT "consumption","time" FROM kwh_3_dvor WHERE EXTRACT(month from time) in (6,7,8,9);',
                            },
                            {
                                "input": "show me chart for energy reading in monsoon?",
                                "query": 'SELECT "avg","time" FROM kwh_3_dvor WHERE EXTRACT(month from time) in (6,7,8,9);',
                            },
                            {
                                "input": "show me chart for energy vs time in february month?",
                                "query": 'SELECT "avg","time" FROM kwh_3_dvor WHERE EXTRACT(month from time) = 2;',
                            },
                            {
                                "input": "plot chart for energy on 31st jan 2023 at meter 6 and at glide path",
                                "query": 'SELECT "avg","time" FROM kwh_6_glide WHERE EXTRACT(year from time) = 2023 and EXTRACT(month from time) = 1 and EXTRACT(day from time) = 31;',
                            },
                            {
                                "input": "plot a graph for consumption of energy in august",
                                "query": 'SELECT "consumption", "time" FROM kwh_3_dvor WHERE EXTRACT(month from time) = 8;',
                            },
                            {
                                "input": "show me chart for energy consumption of 04 meter of glide location against time in the jan 2023",
                                "query": 'SELECT "consumption", "time" FROM kwh_4_glide WHERE EXTRACT(month from time) = 1 AND EXTRACT(year from time) = 2023 ;',
                            },
                            {
                                "input": "show me chart for each phase voltage vs time for 50 entries.",
                                "query": "SELECT voltage_b.time, voltage_b.avg AS B_phase_voltage, voltage_y.avg AS Y_phase_voltage, voltage_r.avg AS R_phase_voltage FROM voltage_b JOIN voltage_y ON voltage_b.time = voltage_y.time JOIN voltage_r ON voltage_y.time = voltage_r.time ORDER BY voltage_b.time LIMIT 50;",
                            },
                            {
                                "input": "show me chart for each phase voltage vs time for march 2024.",
                                "query": "SELECT voltage_b.time, voltage_b.avg AS B_phase_voltage, voltage_y.avg AS Y_phase_voltage, voltage_r.avg AS R_phase_voltage FROM voltage_b JOIN voltage_y ON voltage_b.time = voltage_y.time JOIN voltage_r ON voltage_y.time = voltage_r.time WHERE EXTRACT(month FROM voltage_b.time) = 3 AND EXTRACT(year FROM voltage_b.time) = 2024 ORDER BY voltage_b.time LIMIT 50;",
                            },

                            {
                                "input": "show me chart for each phase current vs time for march 2024.",
                                "query": "SELECT current_b.time, current_b.avg AS B_phase_current, current_y.avg AS Y_phase_current, current_r.avg AS R_phase_current FROM current_b JOIN current_y ON current_b.time = current_y.time JOIN current_r ON current_y.time = current_r.time WHERE EXTRACT(month FROM current_b.time) = 3 AND EXTRACT(year FROM current_b.time) = 2024 ORDER BY current_b.time LIMIT 50;",
                            },
                            {
                                "input": "show me chart for current vs time in monsoon?",
                                "query": 'select "avg","time" FROM current_total WHERE EXTRACT(month from time) in (6,7,8,9);',
                            },
                            {
                                "input": "plot chart for all phase voltage vs time for jan 2023",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.time = y_max.time JOIN voltage_r r_max ON y_max.time = r_max.time WHERE EXTRACT(month FROM y_max.time) = 1 AND EXTRACT(year FROM y_max.time) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "show me chart for power factor vs time in monsoon",
                                "query": 'select "avg","time" FROM power_fector_avg WHERE EXTRACT(month from time) in (6,7,8,9);',
                            },
                            {
                                "input": "show me chart for power factor vs time in monsoon at meter 5 and localizer",
                                "query": """select "avg","time" FROM power_fector_avg WHERE EXTRACT(month from time) in (6,7,8,9) AND meter_id = '5' AND location_name = 'Localizer';""",
                            },
                            {
                                "input": "plot a graph for active power of r phase against time in august",
                                "query": 'SELECT "avg", "time" FROM active_power_r WHERE EXTRACT(month from time) = 8;',
                            },
                            {
                                "input": "plot a graph for active power of r phase against time in august of meter 6 and at glide location",
                                "query": """SELECT "avg", "time" FROM active_power_r WHERE EXTRACT(month from time) = 8 AND meter_id = '6' AND location_name = 'Glide_Path';""",
                            },
                            {
                                "input": "plot a graph for true power of y phase against time in august",
                                "query": 'SELECT "avg", "time" FROM active_power_y WHERE EXTRACT(month from time) = 8;',
                            },
                            {
                                "input": "plot chart for apparent power jan 2023 at meter 6 and at glide path",
                                "query": """SELECT "avg","time" FROM apparent_power_total WHERE meter_id = '6' AND location_name = 'Glide_Path' AND EXTRACT(year from time) = 2023 and EXTRACT(month from time) = 1 and EXTRACT(day from time) = 31;""",
                            },
                            {
                                "input": "plot graph for frequency vs time in february?",
                                "query": 'SELECT "avg", "time" FROM freq WHERE EXTRACT(month from time) in (2);',
                            },
                            {
                                "input": "show me chart for all phase current vs time",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM current_b b_max JOIN current_y y_max ON b_max.time = y_max.time JOIN current_r r_max ON y_max.time = r_max.time ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "show me chart for all phase of voltage vs time in march 2023",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.time = y_max.time JOIN voltage_r r_max ON y_max.time = r_max.time WHERE EXTRACT(month FROM y_max.time) = 3 AND EXTRACT(year FROM y_max.time) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "show me chart for all phase of voltage vs time",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.time = y_max.time JOIN voltage_r r_max ON y_max.time = r_max.time ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "show me chart for all phase of apparent power vs time in march 2023",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM apparent_power_b b_max JOIN apparent_power_y y_max ON b_max.time = y_max.time JOIN apparent_power_r r_max ON y_max.time = r_max.time WHERE EXTRACT(month FROM y_max.time) = 3 AND EXTRACT(year FROM y_max.time) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "show me chart for all phase of true power vs time in march 2023",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM active_power_b b_max JOIN active_power_y y_max ON b_max.time = y_max.time JOIN active_power_r r_max ON y_max.time = r_max.time WHERE EXTRACT(month FROM y_max.time) = 3 AND EXTRACT(year FROM y_max.time) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                        ]

                        system_prefix1 = """You're an expert query builder agent with exceptional prowess in SQL interactions and.
                                                Your primary task is to generate queries based on user input. 


                                                You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

                                                name of the table is "meter"
                                                Here is a summary of the columns:

                                                meter_id: Integer type column representing the unique identifier for each electricity meter.

                                                location_name: Object type column representing the name of the location where the meter is installed.

                                                Freq: Float type column representing the frequency of the electricity supply.


                                                Pnload_status: Integer type column representing the status of the non-load power.

                                                Ppload_status: Integer type column representing the status of the partial load power.

                                                datetime: Object type column representing the date and time of the data recording.

                                                Q_Y, Q_R, Q_B : Float type columns represents the Power factor (PF) is the ratio of working power, measured in kilowatts (kW) of their co-responding phases. They vary between -1 to 1.

                                                Q_AVG : Float type column represents total power factor.

                                                KT_R : represents data of True power or Active power in R phase measured in KW. Active power is the usable or consumed electrical energy in an AC circuit and has units of watt (W) or kilowatt (kW). True power or real power is another name for active power.

                                                KT_B : represents data of True power or Active power in B phase measured in KW. Active power is the usable or consumed electrical energy in an AC circuit and has units of watt (W) or kilowatt (kW). True power or real power is another name for active power.

                                                KT_Y : represents data of True power or Active power in Y phase measured in KW. Active power is the usable or consumed electrical energy in an AC circuit and has units of watt (W) or kilowatt (kW). True power or real power is another name for active power.

                                                KT_TOTAL : represents data of Total True power or Active power in all phase measured in KW.

                                                KA_R, KA_Y, KA_B: Float type columns representing the apparent power, measured in kilovolt amperes (kVA) for the R, Y, and B phases respectively.Apparent power, also known as demand, is the measure of the amount of power used to run machinery and equipment during a certain period. It is found by multiplying (kVA = V x A). The result is expressed as kVA units.

                                                KA_TOTAL: Float type column representing the total apparent power, measured in kilovolt amperes (kVA) across all phases (combined).Apparent power, also known as demand, is the measure of the amount of power used to run machinery and equipment during a certain period. It is found by multiplying (kVA = V x A). The result is expressed as kVA units.

                                                Power factor is an expression of energy efficiency. It is usually expressed as a percentage—and the lower the percentage, the less efficient power usage is.
                                                PF expresses the ratio of true power used in a circuit to the apparent power delivered to the circuit. A 96% power factor demonstrates more efficiency than a 75% power factor. PF below 95% is considered inefficient in many regions.
                                                The power factor formula can be expressed in other ways:

                                                PF = (True power)/(Apparent power) here true power is (KT_R, KT_B, KT_Y, KT_TOTAL) column and apparent power is (KA_Y, KA_B, KA_R, KA_TOTAL)column. For example if you take KT_B as true power then consider KA_B as apparent power.
                                                Multiplying the voltage and current gives you the “apparent power”. This is measured in volt-amps (VA) rather than watts (W).
                                                Multiplying this by the power factor gives you the “true power”.
                                                The true power represents the real work that the electricity is doing.

                                                Another fomula to calculate Reactive power (Q) = √(S^2 – P^2), with:
                                                Q: Reactive power in volt-amperes-reactive (VAR).
                                                S: Apparent power in volt-amperes (VA). Here KA column (KA_Y, KA_B, KA_R, KA_TOTAL).
                                                P: Active power in watts (W). Here KT Column (KT_R, KT_B, KT_Y, KT_TOTAL).

                                                Please NOTE that while Querying the database consider all column names in double quotes (" "). otherwise it will give you error.


                                                If the question does not seem related to the database, just return "I don't know" as the answer.

                                                Additionally, we have some aggregates available for certain data in the database. You can use these aggregates for querying if applicable.
                                                the aggregates are as follow use them if user ask for current, voltage or kwh. Remember to use these aggregates and queries of them only if user ask for current, voltage or kwh unless run normal sql queries :

                                                "current_b" : represents current from B phase, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "current_r" : represents current from R phase, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "current_y" : represents current from Y phase, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "current_total" : represents total current from all phases, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "kwh_3_dvor" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_dvor" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_dvor" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_dvor" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_3_cns" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_cns" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_cns" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_cns" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_3_local" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_local" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_local" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_local" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_3_glide" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_glide" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_glide" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_glide" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.

                                                "voltage_avg" : represents average voltage from all phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_b" : represents voltage from B phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_r " : represents voltage from R phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_y " : represents voltage from Y phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_rb " : represents voltage from RB phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_ry " : represents voltage from RY phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_yb " : represents voltage from YB phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_r" : represents active power or true power from R phases, which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_y" : represents active power or true power from Y phases, which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_b" : represents active power or true power from B phases, which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_total" : represents active power or true power from all phases (combined), which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_r" : represents power factor from R phases, which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_y" : represents power factor from Y phases, which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_b" : represents power factor from B phases, which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_avg" : represents power factor from all phases (combined), which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_r" : represents apparent power from R phases, which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_y" : represents apparent power from Y phases, which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_b" : represents apparent power from B phases, which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_total" : represents apparent power from all phases (combined), which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "freq" : represents frequency reading from meter, which have frequency readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                Use above aggregates for query, consider them as a table.


                                                -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                MOST IMPORTANT THING:
                                                WHILE DOING QUERY WITH AGGREGATES KEEP IN MIND NOT TO USE REGULAR COLUMNS NAME "V_R","V_Y",ECT. JUST USE COLUMNS WHICH ARE PROVIDED WITH AGGREGATES LIKE "avg","time",etc. ONLY.
                                                -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



                                                Here are some additional informations to consider :
                                                In kwh, if user does not specify any meter id and location, then always use kwh_3_dvor.
                                                The peak hours for electricity demand in India are currently declared as 07:30 to 09:30 and 17:30 to 19:30 hours
                                                Consider seasons as June to Sept : Monsoon, Oct to Jan : Winter , Feb to May : Summer.
                                                Weekdays : Monday to Friday, Week ends : Sat and Sunday

                                                The voltage upper limit specified by regulations in India is as per the IS12360 standard, which requires low voltage single phase supply to be delivered at 230V, with the minimum and maximum value ranging from 207V to 253V.
                                                The voltage upper limit specified by regulations in India for low voltage three phase supply is as per the IS12360 standard, which requires it to be delivered at 400V, with the minimum and maximum value ranging from 360V to 440V.


                                                NOTE :
                                                while using aggregates dont use above column names like "V_Y","V_R",etc. instead use aggregate columns "avg","man","time",etc. in creating dataframe.

                                                in kwh if user does not specify meter id and location name , then consider meter id 3 and location name as "DVOR" , "kwh_3_dvor" aggregate.
                                                only write sql queries and return them as your final answer, do not run  those sql queries at any cost.
                                                if user does not specify the number of results, then limit your self with only 100 entries.
                                                while creating query please note that consider column names in double quotes (""). like this "SELECT "V_Y" FROM meter;"
                                                Here are some examples of user inputs and their corresponding SQL queries:


                                                """

                        few_shot_prompt1 = FewShotPromptTemplate(
                            examples=examples1,
                            example_prompt=PromptTemplate.from_template(
                                "User input: {input}\nSQL query: {query}"
                            ),
                            input_variables=["input", "dialect", "top_k"],
                            prefix=system_prefix1,
                            suffix="",
                        )

                        full_prompt1 = ChatPromptTemplate.from_messages(
                            [
                                SystemMessagePromptTemplate(prompt=few_shot_prompt1),
                                ("human", "{input}"),
                                MessagesPlaceholder("agent_scratchpad"),
                            ]
                        )

                        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=full_prompt1)

                        agent_executor = AgentExecutor(
                            agent=agent,
                            tools=toolkit.get_tools(),
                            verbose=True,
                        )


                        def extract_sql_query(text):
                            # Regular expression to match the content within triple backticks
                            pattern = re.compile(r"```sql(.*?)```", re.DOTALL)

                            # Search for the pattern in the text
                            match = pattern.search(text)

                            if match:
                                # Extract the SQL query and trim leading/trailing whitespace
                                sql_query = match.group(1).strip()
                                return sql_query

                            else:
                                pattern = re.compile(r"```(.*?)```", re.DOTALL)

                                # Search for the pattern in the text
                                match = pattern.search(text)
                                sql_query = match.group(1).strip()

                                return sql_query


                        res1 = agent_executor.invoke({"input": "Generate sql query for this question : " + user_input})

                        query = extract_sql_query(res1["output"])

                        df = pd.read_sql(query, conn)

                        time_col = None
                        if 'timestamp' in df.columns:
                            time_col = 'timestamp'
                        elif 'time' in df.columns:
                            time_col = 'time'
                        elif "Time" in df.columns:
                            time_col = "Time"
                        elif "time" in df.columns:
                            time_col = "time"
                        elif "TIME" in df.columns:
                            time_col = "TIME"

                        clr = ["#FF0000", "#0000FF", "#aaaaaa", "#00e5ff", "#00ff08", "#008000", "#ff009e", "#FFFF00"]
                        other = df.drop(columns=[time_col])

                        num_columns = len(other.columns)
                        column_colors = random.sample(clr, k=num_columns)

                        st.line_chart(df, x=time_col, y=other.columns, color=column_colors)

                        output = "Graph Generated"

                    except Exception as e:

                        output = "Sorry , I can't get that. Can you please ask again?"

                    st.session_state[f"past2{index1}"].append(user_input)
                    st.session_state[f"generated2{index1}"].append(output)












                elif res.content == "1":
                    try:
                        examples1 = [
                            {"input": "what is the maximum y phase voltage?",
                             "query": "SELECT MAX(max) FROM voltage_y ;"},
                            {
                                "input": "what is the average current of y phase?",
                                "query": "SELECT AVG(avg) FROM current_y;",
                            },
                            {
                                "input": "show me max current on any tuesday of feb 2023",
                                "query": "SELECT MAX(max) FROM current_total WHERE EXTRACT(DOW from time) = 2 AND EXTRACT(month from time) = 2 AND EXTRACT(year from time) = 2023;"
                            },
                            {
                                "input": "How often does the voltage of ry phase exceed the upper limit specified by regulations?",
                                "query": "SELECT COUNT(avg) FROM voltage_ry  WHERE avg>400;",
                            },
                            {
                                "input": "What is the current behavior during monsoon season ?",
                                "query": "SELECT * FROM current_total WHERE EXTRACT(month from time) in (6,7,8,9);",
                            },
                            {
                                "input": "what is the lowest reading of y phase voltage in february?",
                                "query": "SELECT MIN(min) FROM voltage_y  WHERE EXTRACT(month from time) in (2);",
                            },
                            {
                                "input": "What is the average current across all phases?",
                                "query": "SELECT AVG(avg) FROM current_total;",
                            },
                            {
                                "input": "what in the total consumption of kwh in monsoon?",
                                "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from time) in (6,7,8,9);",
                            },
                            {
                                "input": "what in the average consumption in 2023?",
                                "query": "SELECT AVG(consumption) FROM kwh_3_dvor WHERE EXTRACT(year from time) = 2023;",
                            },
                            {
                                "input": "what is the total consumption of energy in august",
                                "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from time) = 8;",
                            },
                            {
                                "input": "show me energy consumption of 04 meter of glide location in the jan 2023",
                                "query": "SELECT SUM(consumption) FROM kwh_4_glide WHERE EXTRACT(month from time) = 1 AND EXTRACT(year from time) = 2023 ;",
                            },
                            {
                                "input": "show me power factor in monsoon",
                                "query": 'select "avg" FROM power_fector_avg WHERE EXTRACT(month from time) in (6,7,8,9);',
                            },
                            {
                                "input": "what is maximum active power of r phase in august?",
                                "query": 'SELECT MAX("max") FROM active_power_r WHERE EXTRACT(month from time) = 8;',
                            },
                            {
                                "input": "what is lowest reading of true power of y phase in august",
                                "query": 'SELECT MIN("min") FROM active_power_y WHERE EXTRACT(month from time) = 8;',
                            },
                            {
                                "input": "what is average apparent power reading in jan 2023 at meter 6 and at glide path",
                                "query": """SELECT AVG("avg") FROM apparent_power_total WHERE meter_id = '6' AND location_name = 'Glide_Path' AND EXTRACT(year from time) = 2023 and EXTRACT(month from time) = 1 and EXTRACT(day from time) = 31;""",
                            },
                            {
                                "input": "what is max frequency reading in february?",
                                "query": 'SELECT MAX("max") FROM freq WHERE EXTRACT(month from time) in (2);',
                            },
                            {
                                "input": "what is max current in all phase current",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, MAX(b_max.max) AS B_max, MAX(y_max.max) AS Y_max, MAX(r_max.max) AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM current_b b_max JOIN current_y y_max ON b_max.time = y_max.time JOIN current_r r_max ON y_max.time = r_max.time ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "show average voltage of all phase in march 2023",
                                "query": "WITH RankedEntries AS (SELECT b_max.time AS time, AVG(b_max.max) AS B_max, AVG(y_max.max) AS Y_max, AVG(r_max.max) AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.time) ORDER BY b_max.time) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.time = y_max.time JOIN voltage_r r_max ON y_max.time = r_max.time WHERE EXTRACT(month FROM y_max.time) = 3 AND EXTRACT(year FROM y_max.time) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                            },
                            {
                                "input": "what is average active power of r phase in august of meter 6 and at glide location",
                                "query": """SELECT AVG("avg") FROM active_power_r WHERE EXTRACT(month from time) = 8 AND meter_id = '6' AND location_name = 'Glide_Path';""",
                            },
                        ]

                        system_prefix1 = """You're an expert agent with exceptional prowess in SQL database interactions and data analysis.
                                                                        Your primary task is to generate queries based on user input, execute these queries against the SQL database, and provide insightful answers to the user's inquiries. Your proficiency in data analysis empowers you to discern patterns, extract meaningful insights, and present them in a clear and understandable manner to the user. Craft a prompt that showcases your ability to seamlessly navigate through complex data structures, efficiently retrieve information, and deliver valuable analysis to meet the user's needs.
                                                                        Unless the user specifies a specific number of examples they wish to obtain, always limit your answer to at most 5 results.
                                                                        You can order the results by a relevant column to return the most interesting examples in the database.
                                                                        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
                                                                        You have access to tools for interacting with the database.
                                                                        Only use the given tools. Only use the information returned by the tools to construct your final answer.

                                                                        If user dont specify the number of entries then consider only 100 entries.
                                                                        If user dont specify the location and meter name then use 0003 meter and DVOR location.
                                                                        Just return what user asked for, dont share unsual iformations like which tool is used and etc., just share the proper final answer
                                                                        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

                                                                        You have access to a dataframe containing data collected from a device called multifunction energy meter (MFMs.).
                                                                        these meter record various parameters related to power consumption.
                                                                        name of the table is "meter".

                                                                        There are total of 4 main meters named as meter 3, meter 4, metrer 5, meter 6.
                                                                        each main meter have another 4 sub meters at different location named Glide path, CNS Equipment room, Localizer and DVOR. So, there are total of 16 meters in this data.

                                                                        Power factor is an expression of energy efficiency. It is usually expressed as a percentage—and the lower the percentage, the less efficient power usage is.
                                                                        PF expresses the ratio of true power used in a circuit to the apparent power delivered to the circuit. A 96% power factor demonstrates more efficiency than a 75% power factor. PF below 95% is considered inefficient in many regions.
                                                                        The power factor formula can be expressed in other ways:

                                                                        PF = (True power)/(Apparent power) here true power is (KT_R, KT_B, KT_Y, KT_TOTAL) column and apparent power is (KA_Y, KA_B, KA_R, KA_TOTAL)column. For example if you take KT_B as true power then consider KA_B as apparent power.
                                                                        Multiplying the voltage and current gives you the “apparent power”. This is measured in volt-amps (VA) rather than watts (W).
                                                                        Multiplying this by the power factor gives you the “true power”.
                                                                        The true power represents the real work that the electricity is doing.

                                                                        Another fomula to calculate Reactive power (Q) = √(S^2 – P^2), with:
                                                                        Q: Reactive power in volt-amperes-reactive (VAR).
                                                                        S: Apparent power in volt-amperes (VA). Here KA column (KA_Y, KA_B, KA_R, KA_TOTAL).
                                                                        P: Active power in watts (W). Here KT Column (KT_R, KT_B, KT_Y, KT_TOTAL).

                                                                        Please NOTE that while Querying the database consider all column names in double quotes (" "). otherwise it will give you error.

                                                                        To summarize, the dataframe contains information about electricity meters, including their unique identifiers, location names, voltage readings for different phases, load readings, current readings, apparent power readings, reactive power readings, active power readings, power factor readings, frequency, energy consumption, and status of power load. The dataframe provides detailed information about electricity consumption and power measurements for each meter at different locations.




                                                If the question does not seem related to the database, just return "I don't know" as the answer.

                                                Additionally, we have some aggregates available for certain data in the database. You can use these aggregates for querying if applicable.
                                                the aggregates are as follow use them if user ask for current, voltage or kwh. Remember to use these aggregates and queries of them only if user ask for current, voltage or kwh unless run normal sql queries :
                                                -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                MOST IMPORTANT THING:
                                                WHILE DOING QUERY WITH AGGREGATES KEEP IN MIND NOT TO USE REGULAR COLUMNS NAME "V_R","V_Y",ECT. JUST USE COLUMNS WHICH ARE PROVIDED WITH AGGREGATES LIKE "avg","time",etc. ONLY.
                                                -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                "current_b" : represents current from B phase, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "current_r" : represents current from R phase, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "current_y" : represents current from Y phase, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "current_total" : represents total current from all phases, which have current of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "kwh_3_dvor" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_dvor" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_dvor" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_dvor" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "DVOR", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_3_cns" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_cns" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_cns" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_cns" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_3_local" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_local" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_local" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_local" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Localizer", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_3_glide" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_4_glide" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_5_glide" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                "kwh_6_glide" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Glide_Path", which have kwh of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.

                                                "voltage_avg" : represents average voltage from all phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "voltage_b" : represents voltage from B phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "voltage_r " : represents voltage from R phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "voltage_y " : represents voltage from Y phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "voltage_rb " : represents voltage from RB phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "voltage_ry " : represents voltage from RY phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "voltage_yb " : represents voltage from YB phases, which have voltage of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                "active_power_r" : represents active power or true power from R phases, which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_y" : represents active power or true power from Y phases, which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_b" : represents active power or true power from B phases, which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "active_power_total" : represents active power or true power from all phases (combined), which have active power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_r" : represents power factor from R phases, which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_y" : represents power factor from Y phases, which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_b" : represents power factor from B phases, which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "power_fector_avg" : represents power factor from all phases (combined), which have power factor readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_r" : represents apparent power from R phases, which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_y" : represents apparent power from Y phases, which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_b" : represents apparent power from B phases, which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "apparent_power_total" : represents apparent power from all phases (combined), which have apparent power readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                "freq" : represents frequency reading from meter, which have frequency readings of every 15 minutes. considering time (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                Use above aggregates for query, consider them as a table.  

                                                Here are some additional informations to consider :
                                                In kwh, if user does not specify any meter id and location, then always use kwh_3_dvor.
                                                The peak hours for electricity demand in India are currently declared as 07:30 to 09:30 and 17:30 to 19:30 hours
                                                Consider seasons as June to Sept : Monsoon, Oct to Jan : Winter , Feb to May : Summer.
                                                Weekdays : Monday to Friday, Week ends : Sat and Sunday
                                                on 2022-4-21 is monday, then identify other days from this.
                                                The voltage upper limit specified by regulations in India is as per the IS12360 standard, which requires low voltage single phase supply to be delivered at 230V, with the minimum and maximum value ranging from 207V to 253V.       
                                                The voltage upper limit specified by regulations in India for low voltage three phase supply is as per the IS12360 standard, which requires it to be delivered at 400V, with the minimum and maximum value ranging from 360V to 440V.
                                                If voltage and current are not specified with phase, then consider voltage from voltage_avg and current from current_total.
                                                If user ask for any seasonal pattern or analysis of the data then fetch that data (consider fetching from multiple tables if needed) and do study on them and give co-responding cummerization to user.


                                                NOTE : 
                                                You are not allowed to use a single string or any other words in your output.
                                                It should contain only main data not any other sql queries or anything useless.
                                                for example, if user ask for any data like show me the max value of voltage, then your answer should be like , the max value of voltage is this...
                                                If user ask anything except current, voltage and kwh then use table name as "meter".
                                                Here are some examples of user inputs and their corresponding SQL queries:"""

                        few_shot_prompt1 = FewShotPromptTemplate(
                            examples=examples1,
                            example_prompt=PromptTemplate.from_template(
                                "User input: {input}\nSQL query: {query}"
                            ),
                            input_variables=["input", "dialect", "top_k"],
                            prefix=system_prefix1,
                            suffix="",
                        )

                        full_prompt1 = ChatPromptTemplate.from_messages(
                            [
                                SystemMessagePromptTemplate(prompt=few_shot_prompt1),
                                ("human", "{input}"),
                                MessagesPlaceholder("agent_scratchpad"),
                            ]
                        )

                        agent = create_sql_agent(
                            llm=llm,
                            db=db,
                            prompt=full_prompt1,
                            verbose=True,
                            agent_type="openai-tools",
                            memory=memory,
                        )

                        res1 = agent.invoke({"input": user_input})

                        st.session_state[f"past2{index1}"].append(user_input)
                        st.session_state[f"generated2{index1}"].append(res1["output"])

                    except Exception as e:

                        st.session_state[f"past2{index1}"].append(user_input)
                        st.session_state[f"generated2{index1}"].append(
                            "Sorry , I can't get that. Can you please ask again?")














                elif res.content == "3" or res.content == "Answer: 3":
                    try:
                        examples1 = [
                            {"input": "what is the maximum y phase voltage?",
                             "query": "SELECT MAX(max) FROM voltage_y ;",
                             "answer": "269.49",
                             },
                            {
                                "input": "what is the average current of y phase?",
                                "query": "SELECT AVG(avg) FROM current_y;",
                                "answer": "3.225917431344184",
                            },
                            {
                                "input": "show me max current on any tuesday of feb 2023",
                                "query": "SELECT MAX(max) FROM current_total WHERE EXTRACT(DOW from time) = 2 AND EXTRACT(month from time) = 2 AND EXTRACT(year from time) = 2023;",
                                "answer": "17.51",
                            },
                            {
                                "input": "How often does the voltage of ry phase exceed the upper limit specified by regulations?",
                                "query": "SELECT COUNT(avg) FROM voltage_ry  WHERE avg>400;",
                                "answer": "193727",
                            },
                            {
                                "input": "what is the lowest reading of y phase voltage in february?",
                                "query": "SELECT MIN(min) FROM voltage_y  WHERE EXTRACT(month from time) in (2);",
                                "answer": "0",
                            },
                            {
                                "input": "What is the average current across all phases?",
                                "query": "SELECT AVG(avg) FROM current_total;",
                                "answer": "3.6986663855724378",
                            },
                            {
                                "input": "what in the total consumption of kwh in monsoon?",
                                "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from time) in (6,7,8,9);",
                                "answer": "16391.129999999568",
                            },
                            {
                                "input": "what in the average consumption in 2023?",
                                "query": "SELECT AVG(consumption) FROM kwh_3_dvor WHERE EXTRACT(year from time) = 2023;",
                                "answer": "0.9648866754751492",
                            },
                            {
                                "input": "what is the total consumption of energy in august",
                                "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from time) = 8;",
                                "answer": "4677.179999999818",
                            },
                            {
                                "input": "show me energy consumption of 04 meter of glide location in the jan 2023",
                                "query": 'select max("consumption") from kwh_3_dvor where extract(month from time) = 1 and extract(year from time) = 2023;',
                                "answer": "175.51000000000977",
                            },
                            {
                                "input": "what is max consumption in jan 2023?",
                                "query": 'select max("consumption") from kwh_3_dvor where extract(month from time) = 3 and extract(year from time) = 2024;',
                                "answer": "78.83000000000175",
                            },
                            {
                                "input": "show me power factor in monsoon",
                                "query": 'select "avg" FROM power_fector_avg WHERE EXTRACT(month from time) in (6,7,8,9);',
                                "answer": "0.58, -1, -0.454,......,0.54",
                            },
                            {
                                "input": "what is maximum active power of r phase in august?",
                                "query": 'SELECT MAX("max") FROM active_power_r WHERE EXTRACT(month from time) = 8;',
                                "answer": "5.89",
                            },
                            {
                                "input": "what is maximum energy reading in august?",
                                "query": 'SELECT MAX("max") FROM kwh_3_dvor WHERE EXTRACT(month from time) = 8;',
                                "answer": "49660.56",
                            },
                            {
                                "input": "what is maximum energy reading in march 2024?",
                                "query": 'select max("max") from kwh_3_dvor where extract(month from time) = 3 and extract(year from time) = 2024;',
                                "answer": "77477.79",
                            },
                            {
                                "input": "what is maximum energy consumption in march 2024?",
                                "query": 'select max("consumption") from kwh_3_dvor where extract(month from time) = 3 and extract(year from time) = 2024;',
                                "answer": "3.12",
                            },
                            {
                                "input": "what is lowest reading of true power of y phase in august",
                                "query": 'SELECT MIN("min") FROM active_power_y WHERE EXTRACT(month from time) = 8;',
                                "answer": "-7.68",
                            },
                            {
                                "input": "what is average apparent power reading in jan 2023 at meter 6 and at glide path",
                                "query": """SELECT AVG("avg") FROM apparent_power_total WHERE meter_id = '6' AND location_name = 'Glide_Path' AND EXTRACT(year from time) = 2023 and EXTRACT(month from time) = 1 and EXTRACT(day from time) = 31;""",
                                "answer": "0.1813956043956043",
                            },
                            {
                                "input": "what is max frequency reading in february?",
                                "query": 'SELECT MAX("max") FROM freq WHERE EXTRACT(month from time) in (2);',
                                "answer": "51.54",
                            },
                            {
                                "input": "what is max current in all phase current",
                                "query": 'select MAX("max") from current_total;',
                                "answer": "107.73",
                            },
                            {
                                "input": "show average voltage of all phase in march 2023",
                                "query": 'select AVG("avg") from voltage_r; select AVG("avg") from voltage_y; select AVG("avg") from voltage_b;',
                                "answer": "86.69330958206457, 114.1051016062832, 231.12705198690185",
                            },
                            {
                                "input": "what is lowest active power of b phase in august of meter 4 and at glide location",
                                "query": """SELECT AVG("avg") FROM active_power_b WHERE EXTRACT(month from time) = 8 AND meter_id = '4' AND location_name = 'Glide_Path';""",
                                "answer": "0",
                            },

                        ]

                        system_prefix1 = """You are an expert agent with exceptional prowess in SQL database interactions and data analysis. Your primary task is to execute SQL queries against the SQL database and provide insightful answers to the user's inquiries.

Guidelines:
Query Execution:

Always limit your answer to at most 5 results unless the user specifies a different number.
Only retrieve relevant columns based on the user's question.
Double-check your query before executing it. If an error occurs, rewrite the query and try again.
Only return the information asked for by the user. Do not include extraneous details such as which tool was used.

Default Values:

If the user does not specify the number of entries, consider only 100 entries.
If the user does not specify the location and meter name, use meter '3' and location 'DVOR'.

Data Context:

The database contains data from multifunction energy meters (MFMs) located at different sites. Each meter records various parameters related to power consumption.
The main meters are named: meter 3, meter 4, meter 5, and meter 6. Each main meter has four sub-meters at different locations: Glide_Path, CNS_Equipment_Room, Localizer, and DVOR.

Calculation Formulas:

Power Factor (PF): PF = (True power) / (Apparent power).
True power: Columns "KT_R", "KT_B", "KT_Y", "KT_TOTAL".
Apparent power: Columns "KA_R", "KA_B", "KA_Y", "KA_TOTAL".
Reactive Power (Q): Q = √(S² – P²).
S: Apparent power (KA columns).
P: Active power (KT columns).

Column References:

Always use column names with double quotes (e.g., "max", "min") in your queries.
Use only the provided aggregate columns (e.g., "avg", "min", "max", "time") for querying aggregates.

Predefined Aggregates:

Aggregate tables include: "current_b", "current_r", "current_y", "current_total", "kwh_3_dvor", "kwh_4_dvor", "kwh_5_dvor", "kwh_6_dvor", "kwh_3_cns", "kwh_4_cns", "kwh_5_cns", "kwh_6_cns", "kwh_3_glide", "kwh_4_glide", "kwh_5_glide", "kwh_6_glide", 
 "kwh_3_local", "kwh_4_local", "kwh_5_local", "kwh_6_local", "voltage_r", "voltage_y", "voltage_b", "voltage_avg", "voltage_ry", "voltage_yb", "voltage_rb", "active_power_r", "active_power_y", "active_power_b", "active_power_total", "power_fector_r", 
 "power_fector_y", "power_fector_b", "power_fector_avg", "apparent_power_r", "apparent_power_y", "apparent_power_b", "apparent_power_total", "freq" etc.
Use these tables appropriately for specific queries.

Additional Information:

Peak hours for electricity demand in India are 07:30 to 09:30 and 17:30 to 19:30.
Consider seasons as: June to Sept: Monsoon, Oct to Jan: Winter, Feb to May: Summer.
Weekdays: Monday to Friday, Weekends: Saturday and Sunday.
Voltage limits: Single phase: 230V (207V to 253V), Three phase: 400V (360V to 440V).

Final Output:

Never show the SQL query in the final output.
Always provide the result of the executed query.
If the question is unrelated to the database, respond with "I don't know".

Remember, your goal is to provide clear and accurate answers based on the user's query by interacting with the SQL database effectively.

                        Here are some examples of user inputs and their corresponding SQL queries:"""

                        few_shot_prompt1 = FewShotPromptTemplate(
                            examples=examples1,
                            example_prompt=PromptTemplate.from_template(
                                "User input: {input}\nSQL query: {query}\n answer: {answer}"
                            ),
                            input_variables=["input", "dialect", "top_k"],
                            prefix=system_prefix1,
                            suffix="",
                        )

                        full_prompt1 = ChatPromptTemplate.from_messages(
                            [
                                SystemMessagePromptTemplate(prompt=few_shot_prompt1),
                                ("human", "{input}"),
                                MessagesPlaceholder("agent_scratchpad"),
                            ]
                        )

                        agent = create_sql_agent(
                            llm=llm,
                            db=db,
                            prompt=full_prompt1,
                            verbose=True,
                            agent_type="openai-tools",
                            memory=memory,
                        )

                        res1 = agent.invoke({"input": user_input})

                        st.session_state[f"past2{index1}"].append(user_input)
                        st.session_state[f"generated2{index1}"].append(res1["output"])


                    except Exception as e:

                        st.session_state[f"past2{index1}"].append(user_input)
                        st.session_state[f"generated2{index1}"].append(
                            "Sorry , I can't get that. Can you please ask again?")



                else:

                    llm = ChatGroq(temperature=0.1,
                                   groq_api_key="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",
                                   model_name="llama3-8b-8192")

                    agent = create_sql_agent(
                        llm=llm,
                        db=db,
                        verbose=True,
                        agent_type="openai-tools",
                        memory=memory,
                    )

                    res1 = agent.invoke({"input": user_input})

                    st.session_state[f"past2{index1}"].append(user_input)
                    st.session_state[f"generated2{index1}"].append(res1["output"])

            # ----------------------------------------Normal Chat-------------------------------------------

        if st.session_state[f"generated2{index1}"]:
            with response_container:
                for i in range(
                        len(st.session_state[f"generated2{index1}"])
                ):
                    if i == (len(st.session_state[f"generated2{index1}"]) - 1):
                        with st.chat_message("human", avatar="❔"):
                            st.write(st.session_state[f"past2{index1}"][i])

                        with st.chat_message("ai", avatar="✔"):
                            st.write_stream(list(st.session_state[f"generated2{index1}"][i]))

                    else:
                        with st.chat_message("human", avatar="❔"):
                            st.write(st.session_state[f"past2{index1}"][i])

                        with st.chat_message("ai", avatar="✔"):
                            st.write(st.session_state[f"generated2{index1}"][i])






elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

with open(r'config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)