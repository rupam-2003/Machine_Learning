import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["GOOGLE_API_KEY"]=""
llm=ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temparature=0.4)
prompts=ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant"),
    ("human","Question:{Question}")
])
st.title("Longchain demo with Rupam's GPT")
input_text=st.text_input("Enter your question here")
output_parser=StrOutputParser()
chain=prompts|llm|output_parser
if input_text:
    st.write(chain.invoke({'Question':input_text}))
