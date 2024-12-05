from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Artificial intellignece professor. Please respond to the stuent queries."),
        ("user", "Question: {question}")
    ]
)
# Streamlit framework
st.title('Trying Langchain Demo With OPENAI API')
input_text = st.text_input("Welcome to heaven, you will get all your answers")

# OpenAI LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    result = chain.invoke({'question': input_text})
    st.write(result)
