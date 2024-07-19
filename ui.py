import streamlit as st
import time
import requests
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

st.title("GRAVITY CHATBOT 📚")

start = time.time()

if "messages" not in st.session_state:
    st.session_state.messages =  [{"role": "assistant", 
                                "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
history = StreamlitChatMessageHistory(key="chat_messages")

if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        st.session_state["show_expander"] = False
    with st.chat_message("assistant"):
        with st.spinner("문서에서 내용을 찾는 중 입니다."):
            response = requests.post("http://0.0.0.0:819/chat", json={"query" : query})
            if response.status_code==200:
                result = response.json()[0]
                times = response.json()[1]
                print(result)
                st.markdown(result['answer'])
                st.write("GPU 사용시 답변 시간:", time.time() - start)
                st.write("초기 리트리버 구현 :", times['retriever_time'])
                st.write("답변 생성 시간  :", times['chian_invoke_time'])
                st.write("참고 문서 생성 시간 :", times['retriever_invoke_time'])
                with st.expander("참고 문서 확인"):
                    st.text("문서 제목 : " + result["docu_name"])
                    st.text("문서 내용 \n " + result["references"])
    st.session_state.messages.append({"role": "assistant", "content": result['answer']})

print("time :", time.time() - start)