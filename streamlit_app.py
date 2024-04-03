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

load_dotenv()

API_URL = os.getenv('HF_API_URL')
#EMBED_URL = os.getenv('EMBED_URL')

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
    corpus_df = pd.DataFrame(columns=['sentence'])
    
    cursor.execute("SHOW TABLES")
    tables = [row[1] for row in cursor.fetchall()]
    s = f"There are {len(tables)} tables in the schema.\n"
    corpus_df.loc[len(corpus_df.index)] = [s]
    return corpus_df


def semantic_search(model, user_input, corpus_df, context_length):
    
    corpus_embeddings = model.encode(corpus_df['sentence'].tolist(), convert_to_tensor=True)
    input_embed = model.encode(user_input)
    
    hits = util.semantic_search(input_embed, corpus_embeddings, top_k=context_length)
    hits = hits[0]

    context_string = ';'.join([corpus_df['sentence'].iloc[hit['corpus_id']] for hit in hits])
    return context_string


def query(payload):
	response = requests.post(os.getenv('HF_API_URL'), headers=headers, json=payload)
	return response.json()


# Main function to display database information
def main():
    st.title("dRAG - A Database Informed Chatbot :dragon:")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.session_state.embedding_model = model
    
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
            st.session_state.corpus = corpus_df
            
            
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
                    
    #initialize chat message history session state.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    output = query({
                        'inputs': {
                            "context": semantic_search(st.session_state.embedding_model, prompt, st.session_state.corpus, 1),
                            "question": f'{prompt}'
                        }
                    }) 
                    response = output['answer'] 
                    st.write(f':dragon_face:: {response}') 
            message = {"role": "assistant", "content": output}
            # Add assistant response to chat history
            st.session_state.messages.append(message)
        

# Run the main function
if __name__ == "__main__":
    main()
    

