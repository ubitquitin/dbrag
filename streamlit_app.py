import streamlit as st
import snowflake.connector
from databricks import sql
import pandas as pd
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer, util

import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#API_URL = os.getenv('HF_API_URL')
HF_T2S_URL = os.getenv('HF_T2S_URL')

headers = {
	"Accept" : "application/json",
	"Authorization": os.getenv('HF_API_TOKEN'),
	"Content-Type": "application/json" 
}


# Function to connect to Snowflake database
def connect_to_snowflake(user, password, account, database, warehouse, schema):
    
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    return conn

# Function to connect to Databricks database
def connect_to_databricks(hostname, http_path, access_token, catalog, schema):
    
    conn = sql.connect(
        server_hostname = hostname,
        http_path = http_path,
        access_token = access_token
    )
    conn.cursor().execute(f'USE CATALOG {catalog};')
    conn.cursor().execute(f'USE SCHEMA {schema};')
    return conn


def generate_corpus(cursor, database, schema):
    
    st.session_state.gensql_str = ''
    text_data = []
    cursor.execute("SHOW TABLES")
    tables = [row[1] for row in cursor.fetchall()]
    text_data.append(f"There are {len(tables)} tables in the schema.\n")
    table_names = " ".join([i for i in tables])
    text_data.append(f"The names of the tables are {table_names}")

    ### TABLES ###
    for table in tables:
        st.session_state.gensql_str = st.session_state.gensql_str + f'{table} '
        num_cols = 0
        cursor.execute(f"DESCRIBE {table}")
        results = cursor.fetchall()
        pk = ''
        
        for row in results:
            num_cols += 1
            col_name = row[0]
            col_type = row[1]
            is_primary = row[6]
            if is_primary == 'Y':
                pk = col_name

            text_data.append(f"Table {table} has a column {col_name} of type {col_type}.") 
            st.session_state.gensql_str = st.session_state.gensql_str + f'{col_name} {col_type},' 
                 
        st.session_state.gensql_str = st.session_state.gensql_str + f' primary_key: {pk} [SEP]' 
        text_data.append(f"Table {table} has {num_cols} columns.")
        if len(pk) > 0:
            text_data.append(f"{pk} is the primary key of table {table}.")
    
    ### VIEWS ###
    cursor.execute("SHOW VIEWS")
    views = [row[1] for row in cursor.fetchall()]
    text_data.append(f"There are {len(views)} views in the schema.\n")
    view_names = " ".join([i for i in views])
    text_data.append(f"The names of the views are {view_names}")
    
    for view in views:
        st.session_state.gensql_str = st.session_state.gensql_str + f'{view} '
        num_cols = 0
        cursor.execute(f"DESCRIBE {view}")
        results = cursor.fetchall()
        
        for row in results:
            num_cols += 1
            col_name = row[0]
            col_type = row[1]            

            text_data.append(f"View {view} has a column {col_name} of type {col_type}.") 
            st.session_state.gensql_str = st.session_state.gensql_str + f'{col_name} {col_type},' 
                 
        text_data.append(f"View {view} has {num_cols} columns.")
    
    
    ### PROCEDURES ###
    procedure_descriptions = {}
    type_procedures = []
    cursor.execute("SHOW PROCEDURES")
    for row in cursor.fetchall():
        t = f'''The database has a procedure named {row[1]}, that takes in between 
        {row[6]} and {row[7]} arguments.
        '''
        procedure_descriptions[row[1]] = t
    
    query_str = f"""
    SELECT * FROM {database}.INFORMATION_SCHEMA.PROCEDURES 
    WHERE PROCEDURE_CATALOG={database} AND PROCEDURE_SCHEMA={schema};
    """
    cursor.execute(query_str)
    for row in cursor.fetchall():
        if row[2] in procedure_descriptions.values():
            type_procedures.append(row[11])
            procedure_descriptions[row[2]].append(f''' The procedure returns type {row[5]}. 
                                                  The procedure is written in the programming language {row[11]}''')
    
    for key, value in procedure_descriptions.items():
        text_data.append(value)
    lng_counts = Counter(type_procedures)
    
    text_data.append(f'The stored procedures are most commonly written in {lng_counts.most_common(1)}')
    
    ### USER FUNCTIONS ###
    ### COLUMNS ###

    ### LOAD HISTORY ###
    query_str = f"""
    SELECT * FROM {database}.INFORMATION_SCHEMA.LOAD_HISTORY 
    WHERE SCHEMA_NAME={schema} LIMIT 100;
    """
    cursor.execute(query_str)
    load_history_df = cursor.fetch_pandas_all()
    if len(load_history_df) < 100:
        text_data.append(f"There were {len(load_history_df)} files loaded into tables in the past 14 days.")
    else:
        text_data.append("There were 100 or more files loaded into tables in the past 14 days.")   
    
    text_data.append(f"A total of {load_history_df['row_count'].sum()} rows were loaded into tables.")
    text_data.append(f"On average, {load_history_df['row_count'].sum()/len(load_history_df)} rows per file were loaded into tables.")
    text_data.append(f"Of the {len(load_history_df)} files, {len(load_history_df[load_history_df['error_count'] > 0])} files had at least 1 error.")
    
    ### TABLE STORAGE METRICS ###
    query_str = f"""
    SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES 
    WHERE SCHEMA_NAME={schema} LIMIT 100;"""
    
    cursor.execute(query_str)
    tables_df = cursor.fetch_pandas_all()
    text_data.append(f"Autoclustering is enabled for {len(tables_df[tables_df['AUTO_CLUSTERING_ON']==True]/len(tables_df))} percent of your tables.")
    
    text_data.append(f"{len(tables_df[tables_df['IS_TRANSIENT']==True]/len(tables_df))} percent of your tables are transient.")

    
    largest_table_row = tables_df.iloc[tables_df['BYTES'].idxmax()]
    text_data.append(f"""The largest table in the database is {largest_table_row['TABLE_NAME']},
                     with {largest_table_row['BYTES']} bytes over {largest_table_row['ROW_COUNT']} rows.
                     """)
    
    tables_df = tables_df.sort_values(by='BYTES')
    text_data.append(f"Here are the top 10 tables sorted by their size: {tables_df[['NAME', 'BYTES', 'ROW_COUNT']].head(10)}")
    ### COMMANDS (CLUSTERING DEPTH?)###
    
            
    corpus_df = pd.DataFrame(data=text_data, columns=['sentence'])
    return corpus_df


def generate_databricks_corpus(cursor, database, schema):
    
    st.session_state.gensql_str = ''
    text_data = []
    cursor.execute("SHOW TABLES")
    tables = [row[1] for row in cursor.fetchall()]
    text_data.append(f"There are {len(tables)} tables in the schema.\n")
    table_names = " ".join([i for i in tables])
    text_data.append(f"The names of the tables are {table_names}")

    for table in tables:
        st.session_state.gensql_str = st.session_state.gensql_str + f'{table} '
        num_cols = 0
        cursor.execute(f"DESCRIBE {table}")
        results = cursor.fetchall()
        pk = ''
        
        for row in results:
            num_cols += 1
            col_name = row[0]
            col_type = row[1]

            text_data.append(f"Table {table} has a column {col_name} of type {col_type}.") 
            st.session_state.gensql_str = st.session_state.gensql_str + f'{col_name} {col_type},' 
                 
        text_data.append(f"Table {table} has {num_cols} columns.")
            
    corpus_df = pd.DataFrame(data=text_data, columns=['sentence'])
    return corpus_df


def semantic_search(model, user_input, corpus_df, context_length):
    
    corpus_embeddings = model.encode(corpus_df['sentence'].tolist(), convert_to_tensor=True)
    input_embed = model.encode(user_input)
    
    hits = util.semantic_search(input_embed, corpus_embeddings, top_k=context_length)
    hits = hits[0]

    context_string = ';'.join([corpus_df['sentence'].iloc[hit['corpus_id']] for hit in hits])
    return context_string

    
# def query(payload):
# 	response = requests.post(os.getenv('HF_API_URL'), headers=headers, json=payload)
# 	return response.json()


def query_text2sql(payload):
    response = requests.post(os.getenv('HF_T2S_URL'), headers=headers, json=payload)
    return response.json()


# Main function to display database information
def main():
    st.title("dRAG - A Database Informed Chatbot :dragon:")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.session_state.embedding_model = model
    st.session_state.gensql_str = ''
    
    # Database selection
    db_option = st.radio("Select Database:", ("Snowflake", "Databricks"))

    if db_option == "Snowflake":
        st.sidebar.subheader("Snowflake Connection Configuration")
        user = st.sidebar.text_input("User")
        password = st.sidebar.text_input("Password", type="password")
        account = st.sidebar.text_input("Account")
        warehouse = st.sidebar.text_input("Warehouse")
        database = st.sidebar.text_input("Database")
        schema = st.sidebar.text_input("Schema")

        if st.sidebar.button("Connect"):
            conn = connect_to_snowflake(user, password, account, database, warehouse, schema)
            st.success("Connected to Snowflake")

            # Create document corpus and embeddings
            cursor = conn.cursor()
            corpus_df = generate_corpus(cursor, database, schema)  
            st.session_state.cursor = cursor
            st.session_state.corpus = corpus_df
            
            # Wake up huggingface endpoints
            # query({
            #         'inputs': {"question": 'wake up?'}
            #     })
            
            query_text2sql({
                            'inputs': 'Wake up!'
                        }) 
            
            
            
    elif db_option == "Databricks":
        st.sidebar.subheader("Databricks Connection Configuration")
        hostname = st.sidebar.text_input("JDBC URL")
        http_path = st.sidebar.text_input("HTTP Path")
        access_token = st.sidebar.text_input("Access Token", type="password")
        catalog = st.sidebar.text_input("Catalog")
        schema = st.sidebar.text_input("Schema")

        if st.sidebar.button("Connect"):
            conn = connect_to_databricks(hostname, http_path, access_token, catalog, schema)
            st.success("Connected to Databricks")

            # Create document corpus and embeddings
            cursor = conn.cursor()

            corpus_df = generate_databricks_corpus(cursor, database, schema)  
            st.session_state.cursor = cursor
            st.session_state.corpus = corpus_df
            
            # Wake up huggingface endpoints
            # query({
            #         'inputs': {"question": 'wake up?'}
            #     })
            
            query_text2sql({
                            'inputs': 'Wake up!'
                        }) 
    
    intro_str = """
    You are a chatbot designed to help users learn more about their specific Snowflake or Databricks database and schema. 
    You are able to design SQL queries, describe column and table metadata information and provide information about how 
    to optimize their data environments.
    """
    #initialize chat message history session state.
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": intro_str}]
    
    # Display chat messages
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question about your data:"):
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assitant message in chat message container
            # Generate a new response if last message is not from assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                if '!sql' in prompt.lower():
                    #Text2SQL
                    with st.chat_message("assistant", avatar="🐲"):
                        with st.spinner("Thinking..."):
                            output = query_text2sql({
                                'inputs': f'''
                                    "Schema": {st.session_state.gensql_str},
                                    "Question": f'{prompt[4:]}'
                                '''
                            }) 
                            response = output[0]['generated_text'].split('\n')[-1] 
                            st.write(f'I am running this SQL statement in response to your query: \n {response}') 
                            st.session_state.cursor.execute(response)
                            st.write(str(st.session_state.cursor.fetchall()))
                    message = {"role": "assistant", "content": response}
                else:
                    with st.chat_message("assistant", avatar="🐲"):
                        with st.spinner("Thinking..."):
                            context = semantic_search(st.session_state.embedding_model, prompt, st.session_state.corpus, 10)
                            #print(context)
                            # get all but user's prompt
                            injected_prompt_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                            injected_prompt_messages.append({"role": "user", "content": f"Given these facts about the database: {context}. {prompt}"})
                            r = OpenAI().chat.completions.create(
                                messages=injected_prompt_messages, 
                                model="gpt-3.5-turbo",
                            )
                            response = r.choices[0].message.content
                            
                            # output = query({
                            #     'inputs': {
                            #         "context": semantic_search(st.session_state.embedding_model, prompt, st.session_state.corpus, 1),
                            #         "question": f'{prompt}'
                            #     }
                            # }) 
                            #response = output['answer'] 
                            st.write(f'{response}') 
                    message = {"role": "assistant", "content": response}
                # Add assistant response to chat history
                st.session_state.messages.append(message)
        
        except AttributeError as ae:
            st.error('Please use the menu on the left to connect to your database before asking questions!', icon="🚨")
        
        except Exception as e:
            st.error('Something went wrong...', icon="🔥")
        

# Run the main function
if __name__ == "__main__":
    main()
    

