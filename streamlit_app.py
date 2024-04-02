import streamlit as st
import snowflake.connector
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('')
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


def get_embed(text):
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the model output
    embeddings = outputs.last_hidden_state
    
    return embeddings

def generate_corpus(cursor, filename='corpus.csv', corpus_length=100):
    corpus_df = pd.read_csv(filename, sep='\n', names=['sentence', 'embedding'] ,header=False)
    
    cursor.execute("SHOW TABLES")
    tables = [row[1] for row in cursor.fetchall()]
    s = f"There are {len(tables)} tables in the schema.\n"
    corpus_df.loc[len(corpus_df.index)] = [s, get_embed(s)]
    return corpus_df


#Helper function for embedding similarity
def cosine_similarity(a, b):
  dot_product = sum(x * y for x, y in zip(a, b))
  magnitude_a = sum(x * x for x in a)**0.5
  magnitude_b = sum(x * x for x in b)**0.5
  return dot_product / (magnitude_a * magnitude_b)


def semantic_search(user_input, df, context_length):
    
    df['cos_dists'] = df.embedding.apply(lambda x: 1 - cosine_similarity(
    list(map(float,
             x.strip('][').split(', '))), user_input))

    best_inds = df.cos_dists.nlargest(context_length).index.values

    #get text of most similar embedding/
    most_sim_df = df.sentence.iloc[best_inds]
    most_similar_text = '; '.join([i for i in most_sim_df.values])
    context_string = most_similar_text[1:]
    return context_string


def query(payload):
	response = requests.post(os.getenv('HF_API_URL'), headers=headers, json=payload)
	return response.json()


# Main function to display database information
def main():
    st.title("dRAG - A Database Informed Chatbot :dragon_face:")

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
            
            # Q&A
            user_input = st.text_input("Ask a question about your data:")
            
            output = query({
                "context": semantic_search(user_input, corpus_df, 1),
                "question": f'{user_input}',
                "parameters": {}
            })
            
            st.write(f':dragon_face:: {output}')
            
            # cursor.execute("SHOW TABLES")
            # tables = [row[1] for row in cursor.fetchall()]
            # for table in tables:
            #     st.write(f"## Table: {table}")
            #     cursor.execute(f"DESCRIBE TABLE {table}")
            #     columns = cursor.fetchall()
            #     for column in columns:
            #         st.write(f"- {column[0]} | {column[1]}")

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

# Run the main function
if __name__ == "__main__":
    main()
    

