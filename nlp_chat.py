import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import tool
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
file_names = ["1709.07809.txt",
              "ed3bookfeb3_2024.txt",
              "eisenstein-nlp-notes.txt",
              "mml-book.txt",]

file_paths = [os.path.join(current_directory, "Data", file_name) for file_name in file_names]

 
loader = TextLoader(file_paths)
documents = []
for file_path in file_paths:
    loader = TextLoader(file_path)
    documents.extend(loader.load())
dotenv_path = os.path.join(current_directory, ".env")
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")

text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.8)

db.save_local(os.path.join(current_directory, "faiss_index"))
retriever= db.as_retriever(k=4)
from langchain.tools.retriever import create_retriever_tool

def create_system_template():
    SYSTEM_TEMPLATE = """
    Imagine you are a tutor only for the course natural language processing, talk to the user like a tutor who understands their school/college homework problems, if they ask anything else, say 'I dont know'. ":

    <context>
    
    </context>
    """
    return SYSTEM_TEMPLATE

tool = create_retriever_tool(
    retriever,
    "natural_language_processing_database",
    "This is a retreiver tool for tutors to retrieve answers to homeowrk questions in natural language processing course",
)


tools = [tool]
"""
agent chat
"""




#get_word_length.invoke("abc")

llm_with_tools = chat.bind_tools(tools)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
             create_system_template(), 
            
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


agent = create_openai_tools_agent(chat, tools, prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=True)



#agent_executor.invoke({"messages": [HumanMessage(content="suggest some therapists in Boston?")]})

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
)


# Streamlit app

"""
streamlit app
"""
from streamlit_chat import message

st.title("NLP TutorBot")
if 'history' not in st.session_state:
        st.session_state['history'] = []

if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about your homework questions or anything you want to talk about in Natural Language Processing"]

if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Chat:", placeholder="Talk to NLP TutorBot 👉 (:", key='input')
        submit_button = st.form_submit_button(label='Send')
if submit_button and user_input:
    
    response=conversational_agent_executor.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "unused"}},
    )
    output = response['output']

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
          message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="croodles")
    
          message(st.session_state["generated"][i], key=str(i), avatar_style="big-ears-neutral")
          continue

 
response=conversational_agent_executor.invoke(
        {"input": "tell me something else"},
        {"configurable": {"session_id": "unused"}},
    )

end_chat_checkbox = st.checkbox("I have completed interacting with the TutorBot")

end_chat_button = st.button("End Chat")

