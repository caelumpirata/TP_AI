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


memory = ConversationBufferWindowMemory(k=2)

pg_uri = "postgresql+psycopg2://master:0r5VB[TL?>A/8,}<vkpmEwS)@65.20.77.132:32432/ems_ai"
db = SQLDatabase.from_uri(pg_uri)


api = "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"

def plot_data(df):
    # Check the number of columns
    num_columns = len(df.columns)

    # If there are only two columns, plot one against the other
    if num_columns == 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        fig, ax = plt.subplots()
        ax.plot(df[x_col], df[y_col], marker='o')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{y_col} vs {x_col}')
        
        st.pyplot(fig)
        
    # If there are multiple columns, plot each column against a specified column in a single chart
    else:
        # Set the reference column (e.g., 'bucket' or 'timestamp')
        ref_col = 'timestamp' if 'timestamp' in df.columns else 'bucket'
        
        if ref_col not in df.columns:
            st.error("No suitable reference column found (expected 'timestamp' or 'bucket').")
            return
        
        fig, ax = plt.subplots()
        for col in df.columns:
            if col != ref_col:
                ax.plot(df[ref_col], df[col], marker='o', label=col)
        
        ax.set_xlabel(ref_col)
        ax.set_ylabel('Values')
        ax.set_title(f'Columns vs {ref_col}')
        ax.legend()
        
        st.pyplot(fig)



class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])

        return

    def format_plot(self, result):
        # st.write(result)
        st.image(result.get("value", "").split("/")[-1])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


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


def main():
    # Adjust the width of the Streamlit page
    st.set_page_config(page_title="AI Assistant", layout="wide")

    # st.title("AI Assistant")

    if "session" not in st.session_state:
        with open("numbers.txt", "w") as file:
            file.write("0")
        st.session_state["session"] = "ok"

    with open("style.css") as styl:
        st.markdown(f"<style>{styl.read()}</style>", unsafe_allow_html=True)

    Main = st.sidebar.checkbox("Chat With Your data?", key="first")

    if Main:
        st.warning("This feature is in Beta testing, we will Provide it sooner..!")
        # # with open("style.css") as styl:
        # #     st.markdown(f"<style>{styl.read()}</style>", unsafe_allow_html=True)
        # user_option = st.sidebar.selectbox("Choose an option:", ["", "CSV", "SQL"])

        # if user_option == "CSV":

        #     csv = st.sidebar.file_uploader("upload", type="csv", accept_multiple_files=True)

        #     if csv is None:
        #         st.warning("Please upload File....")

        #     csvFiles = []
        #     for file in csv:
        #         csvFiles.append(file.name)

        #     index = 0
        #     csv_file = st.sidebar.selectbox("Select a File", csvFiles)

        #     if csv_file:
        #         for i in range(0, len(csvFiles)):
        #             if csv_file == csvFiles[i]:
        #                 index = i

        #         csv_file = csv[index] if csv else None

        #         if csv_file is None:
        #             st.warning("Please Upload file...")

        #         if csv_file:
        #             st.sidebar.title("Navigation")
        #             selection = st.sidebar.radio("Go to", ["Chat", "EDA"])

        #             # ---------------------------------------------Graph (CSV)----------------------------------------------

        #             if selection == "EDA":

        #                 eda = st.sidebar.selectbox("Choose", ["MITO", "PyGWalker"])

        #                 if eda == "PyGWalker":
        #                     # Save the uploaded CSV file to a temporary location
        #                     temp_csv_path = f"temp_csv{index}.csv"
        #                     with open(temp_csv_path, "wb") as temp_csv_file:
        #                         temp_csv_file.write(csv_file.read())

        #                     try:
        #                         # Try reading the CSV file with 'utf-8' encoding
        #                         df = pd.read_csv(temp_csv_path, encoding="utf-8")
        #                     except UnicodeDecodeError:
        #                         try:
        #                             # Try reading the CSV file with 'latin1' encoding
        #                             df = pd.read_csv(temp_csv_path, encoding="latin1")
        #                         except UnicodeDecodeError:
        #                             # Try reading the CSV file with 'iso-8859-1' encoding
        #                             df = pd.read_csv(temp_csv_path, encoding="iso-8859-1")

        #                     # Show the data in a table
        #                     st.write("Data:")
        #                     st.write(df)

        #                     # Establish communication between pygwalker and streamlit
        #                     init_streamlit_comm()

        #                     # Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of in-process memory.
        #                     @st.cache_resource
        #                     def get_pyg_renderer(df) -> "StreamlitRenderer":
        #                         # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
        #                         return StreamlitRenderer(df, spec="./gw_config.json", debug=False)

        #                     renderer = get_pyg_renderer(df)

        #                     # Render your data exploration interface. Developers can use it to build charts by drag and drop.
        #                     renderer.render_explore()

        #                 if eda == "MITO":
        #                     temp_csv_path = f"temp_csv{index}.csv"
        #                     with open(temp_csv_path, "wb") as temp_csv_file:
        #                         temp_csv_file.write(csv_file.read())

        #                     try:
        #                         # Try reading the CSV file with 'utf-8' encoding
        #                         df = pd.read_csv(temp_csv_path, encoding="utf-8")
        #                     except UnicodeDecodeError:
        #                         try:
        #                             # Try reading the CSV file with 'latin1' encoding
        #                             df = pd.read_csv(temp_csv_path, encoding="latin1")
        #                         except UnicodeDecodeError:
        #                             # Try reading the CSV file with 'iso-8859-1' encoding
        #                             df = pd.read_csv(temp_csv_path, encoding="iso-8859-1")

        #                     final_dfs, code = spreadsheet(df)

        #                     # Display the code that corresponds to the script
        #                     st.code(code)

        #             # -----------------------------------------------Chat (CSV)-------------------------------------------------

        #             elif selection == "Chat":

        #                 with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        #                     tmp_file.write(csv_file.getvalue())
        #                     tmp_file_path = tmp_file.name

        #                 if f"history{index}" not in st.session_state:
        #                     st.session_state[f"history{index}"] = []

        #                 if f"past{index}" not in st.session_state:
        #                     st.session_state[f"past{index}"] = ["Hey ! 👋"]

        #                 if f"generated{index}" not in st.session_state:
        #                     st.session_state[f"generated{index}"] = [

        #                         "Hello !... Ask me anything about " + csv_file.name

        #                     ]

        #                     llm = ChatOpenAI(openai_api_key=api,
        #                                      temperature=0,
        #                                      model="gpt-3.5-turbo-1106")

        #                     def conversational_chat(query, tm):
        #                         agent = create_csv_agent(
        #                             ChatOpenAI(
        #                                 openai_api_key=api,
        #                                 temperature=0,
        #                                 model="gpt-3.5-turbo-1106",
        #                             ),
        #                             path=tm,
        #                             verbose=True,
        #                             agent_type=AgentType.OPENAI_FUNCTIONS,
        #                             memory=memory,
        #                         )
        #                         result = agent.run(query)
        #                         # st.write(agent.agent.llm_chain.prompt.template)
        #                         return result

        #                     user_input = "what this file is about? or explain all columns of this file"

        #                     output = conversational_chat(user_input.lower(), tmp_file_path)

        #                     st.session_state[f"generated{index}"].insert(1, output)
        #                     st.session_state[f"past{index}"].insert(1, user_input)

        #                 # container for the chat history
        #                 response_container = st.container()

        #                 user_input = st.chat_input("How can i assist you?")

        #                 with st.container():

        #                     # -----------------------------------NLP Graphs-----------------------------------------------------

        #                     keys = [
        #                         "graph",
        #                         "chart",
        #                         "plot",
        #                         "chrt",
        #                         "chatr",
        #                         "garph",
        #                         "grph",
        #                     ]
        #                     if user_input:
        #                         for one in keys:
        #                             if one in user_input.lower():
        #                                 try:
        #                                     # Try reading the CSV file with 'utf-8' encoding
        #                                     df = pd.read_csv(
        #                                         tmp_file_path, encoding="utf-8"
        #                                     )
        #                                 except UnicodeDecodeError:

        #                                     try:
        #                                         # Try reading the CSV file with 'latin1' encoding
        #                                         df = pd.read_csv(
        #                                             tmp_file_path, encoding="latin1"
        #                                         )
        #                                     except UnicodeDecodeError:

        #                                         # Try reading the CSV file with 'iso-8859-1' encoding

        #                                         df = pd.read_csv(
        #                                             tmp_file_path, encoding="iso-8859-1"
        #                                         )
        #                                 llm = ChatOpenAI(
        #                                     openai_api_key=api)

        #                                 pand = Agent(
        #                                     df,
        #                                     config={
        #                                         "response_parser": StreamlitResponse,
        #                                         "llm": llm,
        #                                     },
        #                                 )

        #                                 wait_msg = st.warning(
        #                                     "Generating grpah! please wait....."
        #                                 )

        #                                 pand.chat(user_input)
        #                                 wait_msg.empty()
        #                                 output = "Graph generated...."

        #                         # ----------------------------------------Normal Chat-------------------------------------------

        #                         if (
        #                                 "graph" not in user_input.lower()
        #                                 and "chart" not in user_input.lower()
        #                                 and "plot" not in user_input.lower()
        #                         ):
        #                             def conversational_chat(query, tm):
        #                                 agent = create_csv_agent(
        #                                     ChatOpenAI(
        #                                         openai_api_key=api,
        #                                         temperature=0,
        #                                         model="gpt-3.5-turbo-1106",

        #                                     ),
        #                                     path=tm,
        #                                     memory=memory,
        #                                     verbose=True,

        #                                     agent_type=AgentType.OPENAI_FUNCTIONS,
        #                                 )

        #                                 result = agent.run(query)
        #                                 return result

        #                             output = conversational_chat(
        #                                 user_input.lower()
        #                                 .replace("graph", "")
        #                                 .replace("chart", "")
        #                                 .replace("plot", ""),
        #                                 tmp_file_path
        #                             )

        #                         st.session_state[f"past{index}"].append(user_input)
        #                         st.session_state[f"generated{index}"].append(output)

        #                     if st.session_state[f"generated{index}"]:

        #                         with response_container:
        #                             for i in range(
        #                                     len(st.session_state[f"generated{index}"])
        #                             ):
        #                                 message(
        #                                     st.session_state[f"past{index}"][i],
        #                                     is_user=True,
        #                                     key=str(i) + "_user",
        #                                     avatar_style="no-avatar",
        #                                 )
        #                                 message(
        #                                     st.session_state[f"generated{index}"][i],
        #                                     key=str(i),
        #                                     avatar_style="no-avatar",
        #                                 )

        #     else:

        #         st.warning("Please upload file.....!")

        # # --------------------------------------------SQL Section---------------------------------------------------------

        # elif user_option == "SQL":
        #     st.header("SQL Database Connection")

        #     # Streamlit input for database connection details
        #     host = st.text_input("Host:")
        #     user = st.text_input("Username:")
        #     password = st.text_input("Password:", type="password")
        #     database = st.text_input("Database:")
        #     port = st.text_input("port:")

        #     if st.checkbox("Connect"):
        #         if host and user and password and port and database:

        #             conn = create_database_connection(host, user, password, database, port)
        #             st.success("Successfully connected to the database.")

        #             # --------------------------------- To Add New Connection -------------------------------------

        #             get_table_names_query = """SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
        #                                     """

        #             cursor = conn.cursor()
        #             cursor.execute(get_table_names_query)

        #             table_names = cursor.fetchall()
        #             table_names_list = [table_name[0] for table_name in table_names]

        #             table_names_list.insert(0, " ")
        #             table = st.selectbox("Choose table", table_names_list)

        #             if table == " " or table is None:
        #                 st.warning("Please Enter Table name.....")
        #             else:
        #                 data_index = table_names_list.index(table)

        #                 table_connections[table] = conn

        #                 new_database_csv_path = rf"{table}.csv"

        #                 query = f"SELECT * FROM {table}"
        #                 df = pd.read_sql_query(query, conn)
        #                 df.to_csv(new_database_csv_path, index=False, encoding="utf-8")

        #                 st.sidebar.title("TP AI Assistant")

        #                 st.sidebar.title("Navigation")
        #                 selection = st.sidebar.radio("Go to", ["Chat", "EDA"])

        #                 #
        #                 if selection == "Chat":
        #                     # Query the data from the database table
        #                     query = f"SELECT * FROM {table}"
        #                     df = pd.read_sql_query(query, conn)

        #                     tmp_file_paths = rf"{table}.csv"
        #                     df.to_csv(tmp_file_paths, index=False, encoding="utf-8")

        #                     if f"hstry{data_index}" not in st.session_state:
        #                         st.session_state[f"hstry{data_index}"] = []

        #                     if f"generat{data_index}" not in st.session_state:
        #                         st.session_state[f"generat{data_index}"] = [
        #                             "Hello!.. Ask me anything about " + f"{table}"
        #                         ]

        #                     if f"PAST{data_index}" not in st.session_state:
        #                         st.session_state[f"PAST{data_index}"] = ["Hey ! 👋"]

        #                         def conversational_chat(query):
        #                             agent = create_csv_agent(
        #                                 ChatOpenAI(
        #                                     openai_api_key=api,
        #                                     temperature=0,
        #                                     model="gpt-3.5-turbo-1106",
        #                                 ),
        #                                 path=tmp_file_paths,
        #                                 memory=memory,
        #                                 verbose=True,
        #                                 agent_type=AgentType.OPENAI_FUNCTIONS,
        #                             )

        #                             result = agent.run(query)
        #                             return result

        #                         user1 = f"Please explain all columns in in this data."
        #                         output1 = conversational_chat(user1.lower())

        #                         st.session_state[f"PAST{data_index}"].append(user1)
        #                         st.session_state[f"generat{data_index}"].append(output1)

        #                     def conversational_chat(query):
        #                         agent = create_csv_agent(
        #                             ChatOpenAI(
        #                                 openai_api_key=api,
        #                                 temperature=0,
        #                                 model="gpt-3.5-turbo-1106",
        #                             ),
        #                             path=tmp_file_paths,
        #                             memory=memory,
        #                             verbose=True,
        #                             agent_type=AgentType.OPENAI_FUNCTIONS,
        #                         )

        #                         result = agent.run(query)
        #                         return result

        #                     # container for the chat hstry
        #                     response_container = st.container()
        #                     # container for the user's text input
        #                     container = st.container()

        #                     with open("style.css") as styl:
        #                         st.markdown(
        #                             f"<style>{styl.read()}</style>", unsafe_allow_html=True
        #                         )

        #                     with st.form(key="my_form", clear_on_submit=True):
        #                         user_input = st.text_input(
        #                             placeholder="Talk about your database here (:",
        #                             key="input",
        #                         )
        #                         submit_button = st.form_submit_button(label="Send")

        #                     with container:

        #                         if submit_button and user_input:

        #                             # ----------------------------------NLP Graphs (SQL)------------------------------------
        #                             if user_input.lower() != "exit":
        #                                 keys = [
        #                                     "graph",
        #                                     "chart",
        #                                     "plot",
        #                                     "chrt",
        #                                     "chatr",
        #                                     "garph",
        #                                     "grph",
        #                                 ]
        #                                 for one in keys:
        #                                     if one in user_input.lower():
        #                                         llm = OpenAI(openai_api_key=api)
        #                                         pand = Agent(
        #                                             df,
        #                                             config={
        #                                                 "response_parser": StreamlitResponse,
        #                                                 "llm": llm,
        #                                             },
        #                                         )

        #                                         wait_msg = st.warning(
        #                                             "Generating grpah! please wait....."
        #                                         )

        #                                         # st.set_option('deprecation.showPyplotGlobalUse', False)
        #                                         pand.chat(user_input)

        #                                         wait_msg.empty()
        #                                         output = "Graph generated...."

        #                                         # if output:
        #                                         #     os.remove(
        #                                         #         os.path.join(image_folder, latest_image)
        #                                         #     )
        #                                     # ---------------------Normal chat (SQL)----------------------------------------

        #                                 if (
        #                                         "graph" not in user_input.lower()
        #                                         and "chart" not in user_input.lower()
        #                                         and "plot" not in user_input.lower()
        #                                 ):
        #                                     def conversational_chat(query):
        #                                         agent = create_csv_agent(
        #                                             ChatOpenAI(
        #                                                 openai_api_key=api,
        #                                                 temperature=0,
        #                                                 model="gpt-3.5-turbo-1106",
        #                                             ),
        #                                             path=tmp_file_paths,
        #                                             memory=memory,
        #                                             verbose=True,
        #                                             agent_type=AgentType.OPENAI_FUNCTIONS,
        #                                         )

        #                                         result = agent.run(query)
        #                                         return result

        #                                     output = conversational_chat(user_input)

        #                                 st.session_state[f"PAST{data_index}"].append(
        #                                     user_input
        #                                 )
        #                                 st.session_state[f"generat{data_index}"].append(
        #                                     output
        #                                 )
        #                             else:
        #                                 output = "Bye !, have a nice day..."
        #                                 st.session_state[f"PAST{data_index}"].append(
        #                                     user_input
        #                                 )
        #                                 st.session_state[f"generat{data_index}"].append(
        #                                     output
        #                                 )

        #                         if st.session_state[f"generat{data_index}"]:
        #                             with response_container:
        #                                 for i in range(
        #                                         len(st.session_state[f"generat{data_index}"])
        #                                 ):
        #                                     message(
        #                                         st.session_state[f"PAST{data_index}"][i],
        #                                         is_user=True,
        #                                         key=str(i) + "_user",
        #                                         avatar_style="no-avatar",
        #                                     )
        #                                     message(
        #                                         st.session_state[f"generat{data_index}"][i],
        #                                         key=str(i),
        #                                         avatar_style="no-avatar",
        #                                     )

        #                 # -----------------------------------Graph (SQL)--------------------------------------------------

        #                 if selection == "EDA":

        #                     eda = st.sidebar.selectbox("choose", ["MITO", "PyGWalker"])

        #                     if eda == "PyGWalker":
        #                         # Query the data from the database table
        #                         query = f"SELECT * FROM {table}"
        #                         df = pd.read_sql_query(query, conn)

        #                         tmp_file_paths = rf"{table}.csv"
        #                         df.to_csv(tmp_file_paths, index=False, encoding="utf-8")

        #                         # Show the data in a table
        #                         st.write("Table Data:")
        #                         st.write(df)

        #                         # Establish communication between pygwalker and streamlit
        #                         init_streamlit_comm()

        #                         # Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of in-process memory.
        #                         @st.cache_resource
        #                         def get_pyg_renderer(df) -> "StreamlitRenderer":
        #                             # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
        #                             return StreamlitRenderer(df, spec="./gw_config.json", debug=False)

        #                         renderer = get_pyg_renderer(df)

        #                         # Render your data exploration interface. Developers can use it to build charts by drag and drop.
        #                         renderer.render_explore()

        #                     if eda == "MITO":
        #                         # Query the data from the database table
        #                         query = f"SELECT * FROM {table}"
        #                         df = pd.read_sql_query(query, conn)

        #                         tmp_file_paths = rf"{table}.csv"
        #                         df.to_csv(tmp_file_paths, index=False, encoding="utf-8")

        #                         final_dfs, code = spreadsheet(df)

        #                         # Display the code that corresponds to the script
        #                         st.code(code)

        #     else:
        #         st.error("Please fill in all the database connection details.")

    # ---------------------------------------------Graph (CSV)----------------------------------------------
    if Main == False:
        # Button to create a new chat
        add = st.sidebar.button("➕ New Chat")
        st.sidebar.write("Model")
        model1 = st.sidebar.selectbox("Model",
                                      ["llama3-8b-8192", "llama3-70b-8192", " mixtral-8x7b-32768", "gemma-7b-it"])

        if add:
            num = read_numbers_from_file()
            save_number_to_file(int(num[-1]) + 1)

            # st.session_state.index1 += 1
            # st.session_state.chats.append(f"Chat {st.session_state.index1 + 1}")

        numbers = read_numbers_from_file()

        for i in range(1, len(numbers)):
            st.session_state.chats.insert(i - 1, f"chat {i + 1}")
        selected_chat_index = st.sidebar.selectbox("Select Chat", st.session_state.chats)

        # Update index1 based on the selected chat
        index1 = st.session_state.chats.index(selected_chat_index)

        with open("style.css") as styl:
            st.markdown(f"<style>{styl.read()}</style>", unsafe_allow_html=True)

        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", ["Chat", "EDA"], key="navigation_radio")

        if selection == "EDA":
            eda = st.sidebar.selectbox("choose", ["MITO", "PyGWalker"])

            if eda == "PyGWalker":
                # Show the data in a table
                st.write("Data:")
                st.write(dfmain)

                # Establish communication between pygwalker and streamlit
                init_streamlit_comm()

                # Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of in-process memory.
                @st.cache_resource
                def get_pyg_renderer(df) -> "StreamlitRenderer":
                    # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
                    return StreamlitRenderer(df, spec="./gw_config.json", debug=False)

                renderer = get_pyg_renderer(dfmain)

                # Render your data exploration interface. Developers can use it to build charts by drag and drop.
                renderer.render_explore()

            if eda == "MITO":
                final_dfs, code = spreadsheet(dfmain)

                # Display the code that corresponds to the script
                st.code(code)








        # -----------------------------------------------Chat (CSV)-------------------------------------------------

        elif selection == "Chat":

            if f"history2{index1}" not in st.session_state:
                st.session_state[f"history2{index1}"] = []

            if f"past2{index1}" and f"generated2{index1}" not in st.session_state:
                st.title("How can i assist you?")
                st.session_state[f"generated2{index1}"] = []
                st.session_state[f"past2{index1}"] = []

            # container for the chat history
            response_container = st.container()

            user_input = st.chat_input("Chat your here!")
            container = st.container()
            with st.container():

                # -----------------------------------NLP Graphs-----------------------------------------------------

                if user_input:
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
                                   model_name=model1)

                    chain = full_prompt | llm

                    res = chain.invoke({"input": user_input})
                    print(res.content)









                    if res.content == "2" or res.content == "Answer: 2":
                        try:
                            conn = psycopg2.connect(
                                                    host="65.20.77.132", user="master", password="0r5VB[TL?>A/8,}<vkpmEwS)", database="ems_ai", port="32432",
                                                )
                            llm = ChatGroq(temperature=0.1,
                                            groq_api_key="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",
                                            model_name=model1)


                            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                            context = toolkit.get_context()
                            tools = toolkit.get_tools()

                            examples1 = [
                                {
                                    "input": "plot chart of current behavior during monsoon season ?",
                                    "query": 'SELECT "avg", "bucket" FROM current_total WHERE EXTRACT(month from bucket) in (6,7,8,9); ',
                                },
                                {
                                    "input": "plot graph for y phase voltage vs time in february?",
                                    "query": 'SELECT "avg", "bucket" FROM voltage_y WHERE EXTRACT(month from bucket) in (2);',
                                },
                                {
                                    "input": "show me graph for average current across all phases vs time?",
                                    "query": 'SELECT "avg","bucket" FROM current_total LIMIT 100;',
                                },
                                {
                                    "input": "show me chart for consumption of kwh in monsoon?",
                                    "query": 'SELECT "consumption","bucket" FROM kwh_3_dvor WHERE EXTRACT(month from bucket) in (6,7,8,9);',
                                },
                            
                                {
                                    "input": "plot chart for energy on 31st jan 2023 at meter 6 and at glide path",
                                    "query": 'SELECT "avg","bucket" FROM kwh_6_glide WHERE EXTRACT(year from bucket) = 2023 and EXTRACT(month from bucket) = 1 and EXTRACT(day from bucket) = 31;',
                                },
                                {
                                    "input": "plot a graph for consumption of energy in august",
                                    "query": 'SELECT "consumption", "bucket" FROM kwh_3_dvor WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "show me chart for energy consumption of 04 meter of glide location against time in the jan 2023",
                                    "query": 'SELECT "consumption", "bucket" FROM kwh_4_glide WHERE EXTRACT(month from bucket) = 1 AND EXTRACT(year from bucket) = 2023 ;',
                                },
                                
                                {
                                    "input": "show me chart for current vs time in monsoon?",
                                    "query": 'select "avg","bucket" FROM current_total WHERE EXTRACT(month from bucket) in (6,7,8,9);',
                                },
                                {
                                    "input": "plot chart for all phase voltage vs time for jan 2023",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.bucket = y_max.bucket JOIN voltage_r r_max ON y_max.bucket = r_max.bucket WHERE EXTRACT(month FROM y_max.bucket) = 1 AND EXTRACT(year FROM y_max.bucket) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show me chart for power factor vs time in monsoon",
                                    "query": 'select "avg","bucket" FROM power_fector_avg WHERE EXTRACT(month from bucket) in (6,7,8,9);',
                                },
                                {
                                    "input": "show me chart for power factor vs time in monsoon at meter 5 and localizer",
                                    "query": """select "avg","bucket" FROM power_fector_avg WHERE EXTRACT(month from bucket) in (6,7,8,9) AND meter_id = '5' AND location_name = 'Localizer';""",
                                },
                                {
                                    "input": "plot a graph for active power of r phase against time in august",
                                    "query": 'SELECT "avg", "bucket" FROM active_power_r WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "plot a graph for active power of r phase against time in august of meter 6 and at glide location",
                                    "query": """SELECT "avg", "bucket" FROM active_power_r WHERE EXTRACT(month from bucket) = 8 AND meter_id = '6' AND location_name = 'Glide_Path';""",
                                },
                                {
                                    "input": "plot a graph for true power of y phase against time in august",
                                    "query": 'SELECT "avg", "bucket" FROM active_power_y WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "plot chart for apparent power jan 2023 at meter 6 and at glide path",
                                    "query": """SELECT "avg","bucket" FROM apparent_power_total WHERE meter_id = '6' AND location_name = 'Glide_Path' AND EXTRACT(year from bucket) = 2023 and EXTRACT(month from bucket) = 1 and EXTRACT(day from bucket) = 31;""",
                                },
                                {
                                    "input": "plot graph for frequency vs time in february?",
                                    "query": 'SELECT "avg", "bucket" FROM freq WHERE EXTRACT(month from bucket) in (2);',
                                },
                                {
                                    "input": "show me chart for all phase current vs time",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM current_b b_max JOIN current_y y_max ON b_max.bucket = y_max.bucket JOIN current_r r_max ON y_max.bucket = r_max.bucket ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show me chart for all phase of voltage vs time in march 2023",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.bucket = y_max.bucket JOIN voltage_r r_max ON y_max.bucket = r_max.bucket WHERE EXTRACT(month FROM y_max.bucket) = 3 AND EXTRACT(year FROM y_max.bucket) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show me chart for all phase of voltage vs time",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.bucket = y_max.bucket JOIN voltage_r r_max ON y_max.bucket = r_max.bucket ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show me chart for all phase of apparent power vs time in march 2023",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM apparent_power_b b_max JOIN apparent_power_y y_max ON b_max.bucket = y_max.bucket JOIN apparent_power_r r_max ON y_max.bucket = r_max.bucket WHERE EXTRACT(month FROM y_max.bucket) = 3 AND EXTRACT(year FROM y_max.bucket) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show me chart for all phase of true power vs time in march 2023",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, b_max.avg AS B_max, y_max.avg AS Y_max, r_max.avg AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM active_power_b b_max JOIN active_power_y y_max ON b_max.bucket = y_max.bucket JOIN active_power_r r_max ON y_max.bucket = r_max.bucket WHERE EXTRACT(month FROM y_max.bucket) = 3 AND EXTRACT(year FROM y_max.bucket) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
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

                                                    "current_b" : represents current from B phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "current_r" : represents current from R phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "current_y" : represents current from Y phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "current_total" : represents total current from all phases, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "kwh_3_dvor" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_dvor" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_dvor" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_dvor" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_3_cns" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_cns" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_cns" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_cns" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_3_local" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_local" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_local" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_local" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_3_glide" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_glide" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_glide" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_glide" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.

                                                    "voltage_avg" : represents average voltage from all phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_b" : represents voltage from B phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_r " : represents voltage from R phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_y " : represents voltage from Y phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_rb " : represents voltage from RB phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_ry " : represents voltage from RY phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_yb " : represents voltage from YB phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                                                    
                                                    "active_power_r" : represents active power or true power from R phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "active_power_y" : represents active power or true power from Y phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "active_power_b" : represents active power or true power from B phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "active_power_total" : represents active power or true power from all phases (combined), which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "power_fector_r" : represents power factor from R phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                                                    
                                                    "power_fector_y" : represents power factor from Y phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "power_fector_b" : represents power factor from B phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "power_fector_avg" : represents power factor from all phases (combined), which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "apparent_power_r" : represents apparent power from R phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                                                    
                                                    "apparent_power_y" : represents apparent power from Y phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "apparent_power_b" : represents apparent power from B phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "apparent_power_total" : represents apparent power from all phases (combined), which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "freq" : represents frequency reading from meter, which have frequency readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    Use above aggregates for query, consider them as a table.
                                                    

                                                    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                    MOST IMPORTANT THING:
                                                    WHILE DOING QUERY WITH AGGREGATES KEEP IN MIND NOT TO USE REGULAR COLUMNS NAME "V_R","V_Y",ECT. JUST USE COLUMNS WHICH ARE PROVIDED WITH AGGREGATES LIKE "avg","bucket",etc. ONLY.
                                                    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



                                                    Here are some additional informations to consider :
                                                    In kwh, if user does not specify any meter id and location, then always use kwh_3_dvor.
                                                    The peak hours for electricity demand in India are currently declared as 07:30 to 09:30 and 17:30 to 19:30 hours
                                                    Consider seasons as June to Sept : Monsoon, Oct to Jan : Winter , Feb to May : Summer.
                                                    Weekdays : Monday to Friday, Week ends : Sat and Sunday

                                                    The voltage upper limit specified by regulations in India is as per the IS12360 standard, which requires low voltage single phase supply to be delivered at 230V, with the minimum and maximum value ranging from 207V to 253V.
                                                    The voltage upper limit specified by regulations in India for low voltage three phase supply is as per the IS12360 standard, which requires it to be delivered at 400V, with the minimum and maximum value ranging from 360V to 440V.
                                                    

                                                    NOTE :
                                                    while using aggregates dont use above column names like "V_Y","V_R",etc. instead use aggregate columns "avg","man","bucket",etc. in creating dataframe.

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
                            
                            wait_msg = st.warning(
                                            "Generating grpah! please wait....."
                                        )

                            df = pd.read_sql(query, conn)    
                            
                            time_col = None
                            if 'timestamp' in df.columns:
                                time_col = 'timestamp'
                            elif 'bucket' in df.columns:
                                time_col = 'bucket'
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
                                          
                            wait_msg.empty()

                            output = "Graph Generated"

                        except Exception as e:
                            wait_msg.empty()
                            output = "Sorry , I can't get that. Can you please ask again?"

                        st.session_state[f"past2{index1}"].append(user_input)
                        st.session_state[f"generated2{index1}"].append(output)












                    if res.content == "1":
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
                                    "query": "SELECT MAX(max) FROM current_total WHERE EXTRACT(DOW from bucket) = 2 AND EXTRACT(month from bucket) = 2 AND EXTRACT(year from bucket) = 2023;"
                                },
                                {
                                    "input": "How often does the voltage of ry phase exceed the upper limit specified by regulations?",
                                    "query": "SELECT COUNT(avg) FROM voltage_ry  WHERE avg>400;",
                                },
                                {
                                    "input": "What is the current behavior during monsoon season ?",
                                    "query": "SELECT * FROM current_total WHERE EXTRACT(month from bucket) in (6,7,8,9);",
                                },
                                {
                                    "input": "what is the lowest reading of y phase voltage in february?",
                                    "query": "SELECT MIN(min) FROM voltage_y  WHERE EXTRACT(month from bucket) in (2);",
                                },
                                {
                                    "input": "What is the average current across all phases?",
                                    "query": "SELECT AVG(avg) FROM current_total;",
                                },
                                {
                                    "input": "what in the total consumption of kwh in monsoon?",
                                    "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from bucket) in (6,7,8,9);",
                                },
                                {
                                    "input": "what in the average consumption in 2023?",
                                    "query": "SELECT AVG(consumption) FROM kwh_3_dvor WHERE EXTRACT(year from bucket) = 2023;",
                                },
                                {
                                    "input": "what is the total consumption of energy in august",
                                    "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from bucket) = 8;",
                                },
                                {
                                    "input": "show me energy consumption of 04 meter of glide location in the jan 2023",
                                    "query": "SELECT SUM(consumption) FROM kwh_4_glide WHERE EXTRACT(month from bucket) = 1 AND EXTRACT(year from bucket) = 2023 ;",
                                },
                                {
                                    "input": "show me power factor in monsoon",
                                    "query": 'select "avg" FROM power_fector_avg WHERE EXTRACT(month from bucket) in (6,7,8,9);',
                                },
                                {
                                    "input": "what is maximum active power of r phase in august?",
                                    "query": 'SELECT MAX("max") FROM active_power_r WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "what is lowest reading of true power of y phase in august",
                                    "query": 'SELECT MIN("min") FROM active_power_y WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "what is average apparent power reading in jan 2023 at meter 6 and at glide path",
                                    "query": """SELECT AVG("avg") FROM apparent_power_total WHERE meter_id = '6' AND location_name = 'Glide_Path' AND EXTRACT(year from bucket) = 2023 and EXTRACT(month from bucket) = 1 and EXTRACT(day from bucket) = 31;""",
                                },
                                {
                                    "input": "what is max frequency reading in february?",
                                    "query": 'SELECT MAX("max") FROM freq WHERE EXTRACT(month from bucket) in (2);',
                                },
                                {
                                    "input": "what is max current in all phase current",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, MAX(b_max.max) AS B_max, MAX(y_max.max) AS Y_max, MAX(r_max.max) AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM current_b b_max JOIN current_y y_max ON b_max.bucket = y_max.bucket JOIN current_r r_max ON y_max.bucket = r_max.bucket ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show average voltage of all phase in march 2023",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, AVG(b_max.max) AS B_max, AVG(y_max.max) AS Y_max, AVG(r_max.max) AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.bucket = y_max.bucket JOIN voltage_r r_max ON y_max.bucket = r_max.bucket WHERE EXTRACT(month FROM y_max.bucket) = 3 AND EXTRACT(year FROM y_max.bucket) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "what is average active power of r phase in august of meter 6 and at glide location",
                                    "query": """SELECT AVG("avg") FROM active_power_r WHERE EXTRACT(month from bucket) = 8 AND meter_id = '6' AND location_name = 'Glide_Path';""",
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
                                                                            name of the table is "meter"
                                                                            Here is a summary of the columns:

                                                                            meter_id: Integer type column representing the unique identifier for each electricity meter.

                                                                            location_name: Object type column representing the name of the location where the meter is installed.

                                                                            Pnload_status: Integer type column representing the status of the non-load power.

                                                                            Ppload_status: Integer type column representing the status of the partial load power.

                                                                           
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
                                                    WHILE DOING QUERY WITH AGGREGATES KEEP IN MIND NOT TO USE REGULAR COLUMNS NAME "V_R","V_Y",ECT. JUST USE COLUMNS WHICH ARE PROVIDED WITH AGGREGATES LIKE "avg","bucket",etc. ONLY.
                                                    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                    "current_b" : represents current from B phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "current_r" : represents current from R phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "current_y" : represents current from Y phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "current_total" : represents total current from all phases, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "kwh_3_dvor" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_dvor" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_dvor" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_dvor" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_3_cns" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_cns" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_cns" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_cns" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_3_local" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_local" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_local" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_local" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_3_glide" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_4_glide" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_5_glide" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                                                    "kwh_6_glide" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.

                                                    "voltage_avg" : represents average voltage from all phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "voltage_b" : represents voltage from B phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "voltage_r " : represents voltage from R phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "voltage_y " : represents voltage from Y phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "voltage_rb " : represents voltage from RB phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "voltage_ry " : represents voltage from RY phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "voltage_yb " : represents voltage from YB phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                                                    "active_power_r" : represents active power or true power from R phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "active_power_y" : represents active power or true power from Y phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "active_power_b" : represents active power or true power from B phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "active_power_total" : represents active power or true power from all phases (combined), which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "power_fector_r" : represents power factor from R phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                                                    
                                                    "power_fector_y" : represents power factor from Y phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "power_fector_b" : represents power factor from B phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "power_fector_avg" : represents power factor from all phases (combined), which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "apparent_power_r" : represents apparent power from R phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                                                    
                                                    "apparent_power_y" : represents apparent power from Y phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "apparent_power_b" : represents apparent power from B phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "apparent_power_total" : represents apparent power from all phases (combined), which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                                                    "freq" : represents frequency reading from meter, which have frequency readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

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
                            )

                            res1 = agent.invoke({"input": user_input})

                        except Exception as e:
                            res1["ouput"] = "Sorry , I can't get that. Can you please ask again?"

                        st.session_state[f"past2{index1}"].append(user_input)
                        st.session_state[f"generated2{index1}"].append(res1["output"])












                    if res.content == "3" or res.content == "Answer: 3":
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
                                    "query": "SELECT MAX(max) FROM current_total WHERE EXTRACT(DOW from bucket) = 2 AND EXTRACT(month from bucket) = 2 AND EXTRACT(year from bucket) = 2023;"
                                },
                                {
                                    "input": "How often does the voltage of ry phase exceed the upper limit specified by regulations?",
                                    "query": "SELECT COUNT(avg) FROM voltage_ry  WHERE avg>400;",
                                },
                                {
                                    "input": "What is the current behavior during monsoon season ?",
                                    "query": "SELECT * FROM current_total WHERE EXTRACT(month from bucket) in (6,7,8,9);",
                                },
                                {
                                    "input": "what is the lowest reading of y phase voltage in february?",
                                    "query": "SELECT MIN(min) FROM voltage_y  WHERE EXTRACT(month from bucket) in (2);",
                                },
                                {
                                    "input": "What is the average current across all phases?",
                                    "query": "SELECT AVG(avg) FROM current_total;",
                                },
                                {
                                    "input": "what in the total consumption of kwh in monsoon?",
                                    "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from bucket) in (6,7,8,9);",
                                },
                                {
                                    "input": "what in the average consumption in 2023?",
                                    "query": "SELECT AVG(consumption) FROM kwh_3_dvor WHERE EXTRACT(year from bucket) = 2023;",
                                },
                                {
                                    "input": "what is the total consumption of energy in august",
                                    "query": "SELECT SUM(consumption) FROM kwh_3_dvor WHERE EXTRACT(month from bucket) = 8;",
                                },
                                {
                                    "input": "show me energy consumption of 04 meter of glide location in the jan 2023",
                                    "query": "SELECT SUM(consumption) FROM kwh_4_glide WHERE EXTRACT(month from bucket) = 1 AND EXTRACT(year from bucket) = 2023 ;",
                                },
                                
                                {
                                    "input": "show me power factor in monsoon",
                                    "query": 'select "avg" FROM power_fector_avg WHERE EXTRACT(month from bucket) in (6,7,8,9);',
                                },
                                {
                                    "input": "what is maximum active power of r phase in august?",
                                    "query": 'SELECT MAX("max") FROM active_power_r WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "what is lowest reading of true power of y phase in august",
                                    "query": 'SELECT MIN("min") FROM active_power_y WHERE EXTRACT(month from bucket) = 8;',
                                },
                                {
                                    "input": "what is average apparent power reading in jan 2023 at meter 6 and at glide path",
                                    "query": """SELECT AVG("avg") FROM apparent_power_total WHERE meter_id = '6' AND location_name = 'Glide_Path' AND EXTRACT(year from bucket) = 2023 and EXTRACT(month from bucket) = 1 and EXTRACT(day from bucket) = 31;""",
                                },
                                {
                                    "input": "what is max frequency reading in february?",
                                    "query": 'SELECT MAX("max") FROM freq WHERE EXTRACT(month from bucket) in (2);',
                                },
                                {
                                    "input": "what is max current in all phase current",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, MAX(b_max.max) AS B_max, MAX(y_max.max) AS Y_max, MAX(r_max.max) AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM current_b b_max JOIN current_y y_max ON b_max.bucket = y_max.bucket JOIN current_r r_max ON y_max.bucket = r_max.bucket ) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "show average voltage of all phase in march 2023",
                                    "query": "WITH RankedEntries AS (SELECT b_max.bucket AS time, AVG(b_max.max) AS B_max, AVG(y_max.max) AS Y_max, AVG(r_max.max) AS R_max, ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('day', b_max.bucket) ORDER BY b_max.bucket) AS rn FROM voltage_b b_max JOIN voltage_y y_max ON b_max.bucket = y_max.bucket JOIN voltage_r r_max ON y_max.bucket = r_max.bucket WHERE EXTRACT(month FROM y_max.bucket) = 3 AND EXTRACT(year FROM y_max.bucket) = 2023) SELECT time, B_max, Y_max, R_max FROM RankedEntries WHERE rn <= 7 ORDER BY time LIMIT 1000;",
                                },
                                {
                                    "input": "what is lowest active power of b phase in august of meter 4 and at glide location",
                                    "query": """SELECT AVG("avg") FROM active_power_b WHERE EXTRACT(month from bucket) = 8 AND meter_id = '4' AND location_name = 'Glide_Path';""",
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
                                                    name of the table is "meter"
                                                    Here is a summary of the columns:

                                                    meter_id: Integer type column representing the unique identifier for each electricity meter.

                                                    location_name: Object type column representing the name of the location where the meter is installed.

                                                    Pnload_status: Integer type column representing the status of the non-load power.

                                                    Ppload_status: Integer type column representing the status of the partial load power.

                                                    
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
                            

                            "current_b" : represents current from B phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "current_r" : represents current from R phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "current_y" : represents current from Y phase, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "current_total" : represents total current from all phases, which have current of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "kwh_3_dvor" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_4_dvor" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_5_dvor" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_6_dvor" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "DVOR", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_3_cns" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_4_cns" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_5_cns" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_6_cns" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "CNS_Equipment_Room", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_3_local" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_4_local" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_5_local" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_6_local" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Localizer", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_3_glide" : represents energy consumption in kilowatt-hours of meter id 0003 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_4_glide" : represents energy consumption in kilowatt-hours of meter id 0004 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_5_glide" : represents energy consumption in kilowatt-hours of meter id 0005 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.
                            "kwh_6_glide" : represents energy consumption in kilowatt-hours of meter id 0006 and location name as "Glide_Path", which have kwh of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) and consumption (max - min) of every 15 minutes.

                            "voltage_avg" : represents average voltage from all phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "voltage_b" : represents voltage from B phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "voltage_r " : represents voltage from R phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "voltage_y " : represents voltage from Y phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "voltage_rb " : represents voltage from RB phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "voltage_ry " : represents voltage from RY phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "voltage_yb " : represents voltage from YB phases, which have voltage of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum), location_name, meter_id of every 15 minutes.

                            "active_power_r" : represents active power or true power from R phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "active_power_y" : represents active power or true power from Y phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "active_power_b" : represents active power or true power from B phases, which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "active_power_total" : represents active power or true power from all phases (combined), which have active power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "power_fector_r" : represents power factor from R phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                            
                            "power_fector_y" : represents power factor from Y phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "power_fector_b" : represents power factor from B phases, which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "power_fector_avg" : represents power factor from all phases (combined), which have power factor readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "apparent_power_r" : represents apparent power from R phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                            
                            "apparent_power_y" : represents apparent power from Y phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "apparent_power_b" : represents apparent power from B phases, which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "apparent_power_total" : represents apparent power from all phases (combined), which have apparent power readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.

                            "freq" : represents frequency reading from meter, which have frequency readings of every 15 minutes. considering bucket (timestamp), avg (average), min (minimum) , max (maximum) of every 15 minutes.
                            

                            -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                    MOST IMPORTANT THING:
                                                    WHILE DOING QUERY WITH AGGREGATES KEEP IN MIND NOT TO USE REGULAR COLUMNS NAME "V_R","V_Y",ECT. JUST USE COLUMNS WHICH ARE PROVIDED WITH AGGREGATES LIKE "avg","bucket",etc. ONLY.
                            -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
                            )

                            res1 = agent.invoke({"input": user_input})

                        except Exception as e:
                            res1["ouput"] = "Sorry , I can't get that. Can you please ask again?"

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

    # -------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()