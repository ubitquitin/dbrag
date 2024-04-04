# dRAG 🐲
[A chatbot that knows about your specific database!](https://datarag.streamlit.app/)
dRAG uses retrieval augmented generation (RAG) to prompt-inject information about your database when generating responses to your questions!
When you connect to a Snowflake or Databricks Schema using dRAG, it runs a few metadata level queries (low/no cost) to fetch information about the schema structure.

You can then ask dRAG specific questions about your database, and the tables and columns that reside in it!
Additionally, dRAG is capable of answering analyst questions using text2sql technology! dRAG will convert your question into a SQL statement and run the SQL statement in your data warehouse to find the answer for you.


# Technical notes

dRAG uses two huggingface inference endpoints. The first uses a [BERT based model fine tuned on Question-Answer pairs](https://huggingface.co/deepset/roberta-base-squad2) for question answering. The second is a [BERT based model fine tuned on text2sql](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spider) for generating SQL text from a user's textual question.

dRAG generates a corpus of domain specific knowledge about the user's database once the user hits the connect button. This uses the python-snowflake connector (pyspark for databricks) to run some basic low-cost metadata queries, and construct a pandas dataframe of sentences. The sentence-transformer library is used for semantic search, which finds the top N most relevant pieces of textual information to the user's query.
