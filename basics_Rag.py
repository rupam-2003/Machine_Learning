import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
os.environ["GOOGLE_API_KEY"]="" #give your qwn api key
loader=PyPDFLoader(r"D:\\download\\yolov9_paper.pdf")
data=loader.load()
text_spiliter=RecursiveCharacterTextSplitter(chunk_size=1000)
docs=text_spiliter.split_documents(data)
vectorstore=Chroma.from_documents(documents=docs,embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriver=vectorstore.as_retriever(similarity="similar",search_kwargs={"k":10})
llm=ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0
)
st.title("Rupam's Rag application")
query=st.chat_input("Say something")
prompt=query
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt=ChatPromptTemplate.from_messages([
    ("system", system_prompt),
        ("human", "{input}"),
])
if query:
    question_answer_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(retriver,question_answer_chain)
    response=rag_chain.invoke({"input":query})
    st.write(response["answer"])
    
    



