import streamlit as st
import snowflake.connector
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

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

            # Fetch and display tables and columns
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[1] for row in cursor.fetchall()]
            for table in tables:
                st.write(f"## Table: {table}")
                cursor.execute(f"DESCRIBE TABLE {table}")
                columns = cursor.fetchall()
                for column in columns:
                    st.write(f"- {column[0]} | {column[1]}")

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
