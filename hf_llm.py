from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import  AutoTokenizer,AutoModelForCausalLM
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# Define URLs
urls = [
    'https://www.livemint.com/economy/budget-2024-key-highlights-live-updates-nirmala-sitharaman-infrastructure-defence-income-tax-modi-budget-23-july-11721654502862.html',
    'https://cleartax.in/s/budget-2024-highlights',
    'https://www.hindustantimes.com/budget',
    'https://economictimes.indiatimes.com/news/economy/policy/budget-2024-highlights-india-nirmala-sitharaman-capex-fiscal-deficit-tax-slab-key-announcement-in-union-budget-2024-25/articleshow/111942707.cms?from=mdr'
]

# Load and split documents
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create vector store and retriever
vector = Chroma.from_documents(documents=docs, embedding=HuggingFaceEmbeddings())
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the text generation pipeline
model_id = "tiiuae/falcon-7b"
model = AutoModelForCausalLM.from_pretrained(model_id)  # Use AutoModelForCausalLM for PyTorch
tokenizer = AutoTokenizer.from_pretrained(model_id)

text_generation_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define the prompt template
prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Combine the components
llm_chain = prompt | llm | StrOutputParser()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# Streamlit UI
st.title("Rupam's RAG application")
question = st.text_input("Enter something")
if question:
    st.write(rag_chain.invoke(question))
