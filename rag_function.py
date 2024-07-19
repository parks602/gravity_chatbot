from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
import time

llm = ChatOllama(model="samll_chatbot:latest", num_gpu=1 , temperature=0) 

EMBEDDING = HuggingFaceEmbeddings(
    model_name= "BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma(persist_directory="./database/chroma.db", embedding_function=EMBEDDING)

system_prompt = """너는 그라비티 회사의 챗봇이야. 절대 추측하면 안돼. 관련 내용을 찾을 수 없다면 '관련 정보가 없습니다.' 라고만 반드시 이야기해.
너의 역할은 사용자의 질문에 reference를 바탕으로 답변하는거야.
너가 가지고있는 지식은 모두 배제하고, 주어진 reference의 내용만을 바탕으로 답변해야해.
너의 답변은 5줄 이내로 대답해줘.
만약 사용자의 질문이 reference와 관련이 없다면, '제가 가지고 있는 정보로는 답변할 수 없습니다.' 라고만 반드시 말해야해.


Question: {question} 
Context: {context} 
Answer:
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

def getStreamingChain(question: str):
    retriever = vectorstore.as_retriever(search_type = 'similarity', vervose = True)
    chain = (
        {"context": retriever | format_docs, "question":RunnablePassthrough()} 
        | ANSWER_PROMPT 
        | llm 
        | StrOutputParser()
        )
    start = time.time()
    result = chain.invoke(question)
    chian_invoke_time = time.time() - start
    start = time.time()
    context_docs = retriever.invoke(question)
    retriever_invoke_time = time.time() - start
    docu_name = context_docs[0].metadata['source'].split("\\")[1]
    formatted_docs = context_docs[0].page_content

    return {"answer" : result, "docu_name" : docu_name, "references" : formatted_docs}, {"retriever_time" : retriever_time, "chian_invoke_time" : chian_invoke_time, "retriever_invoke_time":retriever_invoke_time}

