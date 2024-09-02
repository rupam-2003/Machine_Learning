from io import BytesIO
import streamlit as  st 
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import os
os.environ["GOOGLE_API_KEY"]="AIzaSyCKWLuWF79YguDT7smZtWC4Elt8EBlar-g"
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)  # Wrap the bytes in a BytesIO object
            num_pages = len(pdf_reader.pages)
            if num_pages == 0:
                st.warning("The PDF appears to have no pages.")
                continue
            st.info(f"Processing {num_pages} pages from the PDF.")
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                    else:
                        st.warning(f"No text could be extracted from page {i + 1}.")
                except Exception as e:
                    st.error(f"Error extracting text from page {i + 1}: {e}")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text
def get_text_chunks(text):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_spliter.split_text(text)
    return chunks
def save_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.info("Vector store saved successfully.")
def load_vector_store():
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("The FAISS index file does not exist. Please ensure it has been created.")
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

    
def get_chain():
    prompt_templates="""
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:

    {context}

    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>

    """
    model=ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)
    prompt=PromptTemplate(template=prompt_templates,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_questions):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_questions)
    chain=get_chain()
    response=chain(
        {"input_documents":docs,"question":user_questions},
        return_only_outputs=True
        
    )
    print(response)
    st.write("Reply: ",response["output_text"])
    
def main():
    st.set_page_config("Rupam's GPT")
    st.header("Chat with Multiple PDFs using Gemini")
    user_question = st.text_input("Ask a question from PDF files")
    if user_question:
        vector_store = load_vector_store()
        if vector_store:
            user_input(user_question)
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your files and click on submit", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit and process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        save_vector_store(text_chunks)
                        st.success("Processing complete.")
                    else:
                        st.warning("No text was extracted from the uploaded PDFs.")
                else:
                    st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()

        
        

    

