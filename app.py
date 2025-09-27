import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="🤖 AI Chatbot (Free)", page_icon="🤖")
st.title("🤖 AI Chatbot with Context Memory (Free)")

# Smaller CPU-friendly model
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_length=256,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=generator)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You: ")

if st.button("Send") and user_input:
    response = conversation.run(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {text}")
    else:
        st.markdown(f"**🤖 {speaker}:** {text}")
