import streamlit as st
from neo4j import GraphDatabase
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.llms import GoogleGenerativeAI

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Google Gemini API
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Initialize LangChain's GoogleGenerativeAI wrapper
llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=1.0)

# Initialize Langchain with Neo4jGraph
try:
    neo4j_graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )

    qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=neo4j_graph,
        verbose=True
    )
except Exception as e:
    st.error(f"Error initializing Neo4jGraph or GraphCypherQAChain: {str(e)}")
    st.stop()

# User Authentication
def create_user(username, password):
    with driver.session() as session:
        result = session.run("CREATE (u:User {username: $username, password: $password}) RETURN u",
                             username=username, password=password)
        return result.single()

def authenticate_user(username, password):
    with driver.session() as session:
        result = session.run("MATCH (u:User {username: $username, password: $password}) RETURN u",
                             username=username, password=password)
        return result.single()

# Function to handle AI conversation
def ai_diagnosis_interview(user_info, chat_history):
    try:
        response = qa_chain.run(user_info)
        chat_history.append((user_info, response))
        return response, chat_history
    except Exception as e:
        st.error(f"Error in AI interview: {str(e)}")
        return "I'm sorry, but I encountered an error processing your request.", chat_history

# Streamlit app
st.title("AI-Powered Ayurvedic App")

menu = ["Home", "Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Home")
    st.write("Welcome to the AI-Powered Ayurvedic App!")

elif choice == "Login":
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.success(f"Welcome {username}")
            user_info = f"User info: {dict(user['u'])}"

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Proceed to AI diagnosis interview
            diagnosis, st.session_state.chat_history = ai_diagnosis_interview(user_info, st.session_state.chat_history)
            st.write("Diagnosis and Treatment Plan:")
            st.write(diagnosis)

            # Continue conversation
            user_response = st.text_input("Your Response", key="user_response")
            if st.button("Submit", key="submit_response"):
                diagnosis, st.session_state.chat_history = ai_diagnosis_interview(user_response, st.session_state.chat_history)
                st.write(diagnosis)

        else:
            st.error("Invalid username or password")

elif choice == "Sign Up":
    st.subheader("Sign Up")

    new_username = st.text_input("Create Username")
    new_password = st.text_input("Create Password", type="password")

    if st.button("Sign Up"):
        user = create_user(new_username, new_password)
        if user:
            st.success(f"Account created successfully for {new_username}")
        else:
            st.error("Error creating account")

# Close the Neo4j driver connection on app exit
def on_exit():
    driver.close()

st.on_session_end(on_exit)