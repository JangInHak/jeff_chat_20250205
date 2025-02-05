import streamlit as st
import os
from dotenv import load_dotenv
from dotenv import dotenv_values

import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_anthropic import ChatAnthropic

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

model = "gpt-4o-mini"
llm = ChatOpenAI(model_name=model, temperature=0)

template = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

st.header("RAG based Chat with PDF")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



def main():

    with st.sidebar:
        openAI_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        os.environ["OPENAI_API_KEY"] = openAI_api_key

        "[Get an OpenAI API KEY]"

    if prompt := st.chat_input():
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
        
    
        # 디스크에서 Vector DB를 로드합니다.
        # FAISS 불러오기
        new_db_faiss = FAISS.load_local("fall_faiss_index_20250205", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db_faiss.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.7}            
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | template
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(prompt)

        msg = response

        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)



if __name__ == "__main__":
    main()
    
