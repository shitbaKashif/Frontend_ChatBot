# import os
# import csv
# from langchain.llms import Replicate
# from langchain.vectorstores import FAISS
# from langchain import PromptTemplate, LLMChain
# from langchain.chains import ConversationChain
# from langchain.embeddings import EmbaasEmbeddings
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains.question_answering import load_qa_chain


# EMBAAS_API_KEY = '8695090BC2FB9B89ACE98697DCD87BB8B6A11591C96D3C9A05ABB5D431948D97'
# REPLICATE_API_TOKEN = 'r8_IxEKEXoHTlfbepMNFN7XUi8GM21Tq6j1qz4Jg'
# os.environ["EMBAAS_API_KEY"] = EMBAAS_API_KEY
# os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# import requests
# import xml.etree.ElementTree as ET

# url = "https://www.nu.edu.pk/sitemap.xml"

# # Fetch the XML data from the URL
# response = requests.get(url)
# data = response.text

# # Parse the XML data
# root = ET.fromstring(data)

# # Extract the <loc> part and append it to an array
# urls_array = []
# for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
#     urls_array.append(url.text)

# # Remove the problematic URL if it exists in the list
# urls_array = [url for url in urls_array if url != 'http://nu.edu.pk/Campus']

# urls = list(urls_array)

# all_data = []

# for url in urls:
#     try:
#         loader = WebBaseLoader(url)
#         data = loader.load()
#         all_data.extend(data)
#     except Exception as e:
#         print(f"Failed to scrape {url}: {e}")

# print(type(all_data),len(all_data))


# import re
# def clean_page_content(page_content):
#     # Remove any leading or trailing whitespace and newlines
#     cleaned_page_content = page_content.strip()
#     # Remove extra newlines and whitespaces within the content
#     cleaned_page_content = re.sub(r'\n+', '\n', cleaned_page_content)
#     # Remove any extra spaces at the beginning of each line
#     cleaned_page_content = re.sub(r'^\s+', '', cleaned_page_content, flags=re.MULTILINE)
#     # Remove "Home" and "Contact" from the content
#     cleaned_page_content = re.sub(r'\b(?:Home|Contact)\b', '', cleaned_page_content)
#     return cleaned_page_content

# def clean_dataset(data):
#     # Initialize an empty list to store the cleaned page content for each document
#     cleaned_data = []

#     # Iterate over each document in the dataset
#     for doc in data:
#         # Get the page_content from the document
#         page_content = doc.page_content

#         # Clean the page_content using the clean_page_content function
#         cleaned_page_content = clean_page_content(page_content)

#         # Append the cleaned page_content to the cleaned_data list
#         cleaned_data.append(cleaned_page_content)

#     return cleaned_data

# import textwrap

# def wrap_text_preserve_newlines(text, width=110):
#     # Split the input text into lines based on newline characters
#     lines = text.split('\n')

#     # Wrap each line individually
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

#     # Join the wrapped lines back together using newline characters
#     wrapped_text = '\n'.join(wrapped_lines)

#     return wrapped_text

# text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
# docs = text_splitter.split_documents(all_data)
# print(len(docs))

# from langchain.embeddings import TfidfEmbeddings
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer()

# # Assuming you have the 'docs' list of Document objects
# # Create the list of page content from 'docs'
# documents = [doc.page_content for doc in docs]

# # Fit the TfidfVectorizer on the documents and get embeddings
# doc_embeddings = tfidf_vectorizer.fit_transform(documents).toarray()

# # Create the FAISS index from the embeddings
# db = FAISS.from_embeddings(doc_embeddings)

# llm = Replicate(
#     model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
#     input={"temperature": 0.75, "max_length": 500, "top_p": 1},)
# chain = load_qa_chain(llm, chain_type="stuff")

# query = "State your Fee Refund Policy?"
# docs = db.similarity_search(query)
# Res = chain.run(input_documents=docs, question=query)
# # print(wrap_text_preserve_newlines(str(Res)))
# # Res

# # Create a ConversationBufferWindowMemory to store the conversation history
# mem = ConversationBufferWindowMemory(k=1000)
# conversation = ConversationChain(llm=llm, verbose=True, memory=mem)

# # Function to save student details to a text file
# def save_student_details(name, mobile_number, degree):
#     with open("student_details.txt", "w") as file:
#         file.write(f"Name: {name}\n")
#         file.write(f"Mobile Number: {mobile_number}\n")
#         file.write(f"Degree: {degree}\n")

# # Function to get similar documents using similarity search
# def get_similar_documents(query):
#     docs = db.similarity_search(query)
#     return docs

# def get_chatbot_response(question, docs):
#     # Get the conversation history from the conversation object
#     conversation_history = []
#     for item in conversation.memory:
#         if "input" in item:
#             conversation_history.append(f"You: {item['input']}")
#         if "response" in item:
#             conversation_history.append(f"Chatbot: {item['response']['message']}")
#     response = chain.run(input_documents=docs, question=question)
#     # Return the response message and conversation history as a dictionary
#     return {"message": response, "conversation": conversation_history}

# def run_conversation():
#     print("Chatbot: Hi there! I'm an AI. What's your name?")
#     name = input("You: ")

#     print("Chatbot: Nice to meet you, " + name + "! What's your mobile number?")
#     mobile_number = input("You: ")

#     print("Chatbot: Great! Which degree do you want to apply for?")
#     degree = input("You: ")

#     save_student_details(name, mobile_number, degree)
#     print("Chatbot: Thank you for providing your details. How can I help you?")

#     while True:
#         user_input = input("You: ").strip().lower()
#         # conversation_history.append(f"You: {user_input}")

#         if user_input == 'q':
#             print("Chatbot: Goodbye! Have a great day!")
#             break

#         similar = get_similar_documents(user_input)
#         response = get_chatbot_response(user_input, similar)
#         # conversation_history.append(f"Chatbot: {response}")
#         print("Chatbot: ", response)

# # Run the conversation
# run_conversation()


import os
import csv
import textwrap
from langchain.chains import ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory

# Load Language Model (LLM) using HuggingFaceEmbeddings
model_name = "gpt2"  # Replace with your desired language model
embeddings = HuggingFaceEmbeddings(model_name)

# Create the Conversation Chain with Long-Term Memory
mem = ConversationBufferWindowMemory(k=1000)  # Long-term memory with a buffer size of 1000
chain = ConversationChain(llm=embeddings, verbose=False, memory=mem)

# Function for prompt engineering
def add_prompt(input_text):
    # Customized prompt to inform the chatbot about its purpose and long-term memory
    prompt = """
    User: {0}
    Chatbot: Hi there! I'm an AI chatbot here to help you with university admission-related queries using the university's official content. I have a long-term memory, so I'll remember the first question and its answer even after 1000 questions. Please feel free to ask any questions, and I'll do my best to provide you with accurate information.
    """.format(input_text)
    return prompt

# Function to save student details to a text file
def save_student_details(name, mobile_number, degree):
    with open("student_details.txt", "w") as file:
        file.write(f"Name: {name}\n")
        file.write(f"Mobile Number: {mobile_number}\n")
        file.write(f"Degree: {degree}\n")

# Main chatbot function
def university_admission_chatbot():
    print("Chatbot: Hi there! I'm an AI chatbot here to help you with university admission-related queries.")
    print("Chatbot: Please feel free to ask any questions, and I'll do my best to provide you with the information you need.")
    
    name, mobile_number, degree = None, None, None

    # Check if the chatbot is collecting student details
    if name is None:
        print("Chatbot: What's your name?")
        name = input("You: ").strip().title()

    if mobile_number is None:
        print("Chatbot: Nice to meet you, " + name + "! What's your mobile number?")
        mobile_number = input("You: ").strip()

    if degree is None:
        print("Chatbot: Great! Which degree do you want to apply for?")
        degree = input("You: ").strip().title()
        # Save student details to text file
        save_student_details(name, mobile_number, degree)
        print("Chatbot: Thank you for providing your details. How can I assist you further?")
        
    while True:
        user_input = input("You: ").strip()
        
        # Exit condition
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Add prompt engineering to user input
        input_text_with_prompt = add_prompt(user_input)

        # Generate response using the Conversation Chain with LLM
        response = chain.run(input_text=input_text_with_prompt)
        print("Chatbot:", response)

# Run the university admission chatbot
university_admission_chatbot()
