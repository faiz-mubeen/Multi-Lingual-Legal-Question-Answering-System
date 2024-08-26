import os
# Setting the OpenAI API key as an environment variable for authentication
os.environ['OPENAI_API_KEY']='sk-proj-plBrqW3ezb9i0LXu535aT3BlbkFJMfcVABBZn9wgJzPh1TQo'

# Importing necessary libraries from LangChain and Streamlit
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
import streamlit as st

# Importing Google Translator for translating outputs to different languages
from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='auto', target='ur')

# Setting the directory where Chroma will store its persistent data
persist_directory='ip+pdf'

# Initializing OpenAI Embeddings for vectorization of text
embedding = OpenAIEmbeddings()

# Initializing Chroma vector database, with embeddings and persistence setup
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Importing and initializing Ollama model (an alternative to OpenAI)
from langchain_community.llms import Ollama

# Creating an instance of OpenAI LLM with temperature set to 0.3 for more deterministic responses
llm = OpenAI(temperature=0.3)

# Defining a prompt template for the chatbot which guides its behavior and ensures accurate legal responses
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Given the following context and a question, generate an answer based on this context only.
You're a chatbot specializing in Indian law, particularly the Indian Penal Code (IPC). Your user seeks guidance on legal matters within the scope of the IPC.
Engage with them in a helpful and informative manner, providing accurate answers to their questions and clarifications regarding relevant legal concepts and statutes.
Ensure your responses are clear, concise, and tailored to the specific queries posed.
In the answer, try to provide as much text as possible from the source document context.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
<context>
{context}
</context>
Question: {input}
""")

# Creating a document chain to combine documents with the legal question answering model
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Setting up a retriever that uses the Chroma vector database
retriever = vectordb.as_retriever()

# Creating a retrieval chain that connects the retriever and the document chain
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Setting up the Streamlit interface title
st.title("LegalEase")

# Initializing Google Translator again to set default target language as Urdu
from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='auto', target='ur')

# Function to display chat messages in the Streamlit chat UI
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                for i in range(0, len(message["images"]), NUM_IMAGES_PER_ROW):
                    cols = st.columns(NUM_IMAGES_PER_ROW)
                    for j in range(NUM_IMAGES_PER_ROW):
                        if i + j < len(message["images"]):
                            cols[j].image(message["images"][i + j], width=200)

# Creating a sidebar with a title and brief description about LegalEase
with st.sidebar:
    st.title("𓍝 Legal chat")
    st.subheader("About LegalEase")
    st.markdown(
        """LegalEase aims to simplify access to legal knowledge by creating a user-friendly system where individuals can ask
          legal questions and get answers in everyday language, regardless of their level of expertise in law."""
    )
    st.header("Language")
    st.success("Choose the language of your answer", icon="💚")

# Sidebar for language selection with radio buttons
with st.sidebar:
    mode = st.radio(
        "", options=["en","hi","ur","ta"], captions=["English","Hindi", "Urdu", "Tamil"]
    )

# Adding a divider in the UI
st.divider()

# Initializing session state for messages and greetings if they don't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Function call to display previously stored chat messages from session history
display_chat_messages()

# Greet the user upon first visit with a welcome message
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Hello there! I'm LegalEase, your friendly assistant for navigating the legal realm with ease. Whether you're deciphering legal documents, seeking advice on rights and regulations, or exploring legal procedures, I'm here to guide you every step of the way. Let's dive into the world of law together!"
        st.markdown(intro)
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

# Providing some example prompts for user interaction
example_prompts = [
    "What are the legal implications of a hit-and-run accident?",
    "How can I claim compensation for a workplace injury?",
    "What are the legal options for resolving a property dispute with a family member?",
    "What are the legal requirements for making a will in India?",
    "How can I file a complaint against online fraud or cybercrime?",
    "What are the legal rights and procedures for adoption in India?"
]

# Providing helpful tooltips for example prompts
example_prompts_help = [
    "What happens if someone hits a car and drives away without stopping?",
    "How can I get money if I get hurt while working?",
    "What can I do if there's a fight over property with someone in my family?",
    "What do I need to do legally to write down who gets my things when I'm gone in India?",
    "What should I do if someone tricks me online or does something bad to me using the internet?",
    "How can I adopt a child legally in India?"
]

# Creating two rows of three columns each to display example question buttons
button_cols = st.columns(3)
button_cols_2 = st.columns(3)

# Variable to track which example question button was pressed
button_pressed = ""

# Adding buttons for the example prompts in the first row
if button_cols[0].button(example_prompts[0], help=example_prompts_help[0]):
    button_pressed = example_prompts[0]
elif button_cols[1].button(example_prompts[1], help=example_prompts_help[1]):
    button_pressed = example_prompts[1]
elif button_cols[2].button(example_prompts[2], help=example_prompts_help[2]):
    button_pressed = example_prompts[2]

# Adding buttons for the example prompts in the second row
elif button_cols_2[0].button(example_prompts[3], help=example_prompts_help[3]):
    button_pressed = example_prompts[3]
elif button_cols_2[1].button(example_prompts[4], help=example_prompts_help[4]):
    button_pressed = example_prompts[4]
elif button_cols_2[2].button(example_prompts[5], help=example_prompts_help[5]):
    button_pressed = example_prompts[5]

# Main user input area; either accept input from chat box or via button press
if prompt := (st.chat_input("Please type your legal question or concern here.") or button_pressed):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    prompt = prompt.replace('"', "").replace("'", "")

    # If there's a valid prompt, generate an answer using the retrieval chain
    if prompt != "":
        response1 = retrieval_chain.invoke({"input": prompt})
        
        # Translate the response if a language other than English is selected
        if mode != 'en':
            translator = GoogleTranslator(source='auto', target=mode)
            df = translator.translate(response1["answer"])
        else:
            df = response1["answer"]
        
        response = ""
        
        # Display the assistant's response in the chat
        with st.chat_message("assistant"):
            full_response = df
            message_placeholder = st.empty()
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            response += full_response + " "

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        # Rerun the app to refresh the interface
        st.experimental_rerun()
