import streamlit as st
import snowflake.connector
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pandas as pd
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
def connect_to_databricks(jdbc_url, user, password, driver):
    spark = SparkSession.builder \
        .appName("Databricks Streamlit App") \
        .config("spark.driver.extraClassPath", driver) \
        .getOrCreate()

    properties = {
        "user": user,
        "password": password
    }

    sql_context = SQLContext(spark)
    return spark, sql_context


def generate_corpus(cursor, corpus_length=100):
    
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
            is_primary = row[6]
            if is_primary == 'Y':
                pk = col_name

            text_data.append(f"Table {table} has a column {col_name} of type {col_type}.") 
            st.session_state.gensql_str = st.session_state.gensql_str + f'{col_name} {col_type},' 
                 
        st.session_state.gensql_str = st.session_state.gensql_str + f' primary_key: {pk} [SEP]' 
        text_data.append(f"Table {table} has {num_cols} columns.")
        if len(pk) > 0:
            text_data.append(f"{pk} is the primary key of table {table}.")
            
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
            corpus_df = generate_corpus(cursor)  
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
        jdbc_url = st.sidebar.text_input("JDBC URL")
        user = st.sidebar.text_input("User")
        password = st.sidebar.text_input("Password", type="password")
        driver = st.sidebar.text_input("JDBC Driver Class Name")

        if st.sidebar.button("Connect"):
            spark, sql_context = connect_to_databricks(jdbc_url, user, password, driver)
            st.success("Connected to Databricks")

            # Fetch and display tables and columns
            tables = sql_context.read.jdbc(jdbc_url, "SHOW TABLES", properties={"user": user, "password": password})
            table_names = [row['tableName'] for row in tables.collect()]

            for table in table_names:
                st.write(f"## Table: {table}")
                columns = sql_context.read.jdbc(jdbc_url, f"DESCRIBE TABLE {table}", properties={"user": user, "password": password})
                for row in columns.collect():
                    st.write(f"- {row['col_name']} | {row['data_type']}")
    
    intro_str = """
    You are a chatbot designed to help users learn more about their specific Snowflake or Databricks database and schema. 
    You are able to design SQL queries, describe column and table metadata information and provide information about how 
    to optimize their data environments. To start, tell the user one or two things they can ask you to learn more.
    """
    #initialize chat message history session state.
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": intro_str}]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question about your data:"):
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
                        context = semantic_search(st.session_state.embedding_model, prompt, st.session_state.corpus, 5)
                        # get all but user's prompt
                        injected_prompt_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                        injected_prompt_messages.append({"role": "user", "content": f"Given {context}, {prompt}"})
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
        

# Run the main function
if __name__ == "__main__":
    main()
    

