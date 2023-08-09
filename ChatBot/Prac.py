# import os
# import re
# import textwrap
# import requests
# import xml.etree.ElementTree as ET
# from langchain.vectorstores import FAISS
# from langchain import PromptTemplate, LLMChain
# from langchain.chains import ConversationChain
# from langchain.document_loaders import WebBaseLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains.question_answering import load_qa_chain



# os.environ["OPENAI_API_KEY"] = "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"

# # Fetch URLs from sitemap
# url = "https://www.nu.edu.pk/sitemap.xml"
# response = requests.get(url)
# data = response.text
# root = ET.fromstring(data)
# urls_array = [url.text for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
# urls = [url for url in urls_array if url != 'http://nu.edu.pk/Campus']
# all_data = []

# # Scrape data from URLs
# for url in urls:
#     try:
#         loader = WebBaseLoader(url)
#         data = loader.load()
#         all_data.extend(data)
#     except Exception as e:
#         print(f"Failed to scrape {url}: {e}")

# # Clean page content
# def clean_page_content(page_content):
#     cleaned_page_content = page_content.strip()
#     cleaned_page_content = re.sub(r'\n+', '\n', cleaned_page_content)
#     cleaned_page_content = re.sub(r'^\s+', '', cleaned_page_content, flags=re.MULTILINE)
#     cleaned_page_content = re.sub(r'\b(?:Home|Contact)\b', '', cleaned_page_content)
#     return cleaned_page_content

# # Clean dataset
# def clean_dataset(data):
#     cleaned_data = []
#     for doc in data:
#         page_content = doc.page_content
#         cleaned_page_content = clean_page_content(page_content)
#         cleaned_data.append(cleaned_page_content)
#     return cleaned_data

# # Wrap text with newlines
# def wrap_text_preserve_newlines(text, width=110):
#     lines = text.split('\n')
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
#     wrapped_text = '\n'.join(wrapped_lines)
#     return wrapped_text

# def add_prompt(input_text):
#     prompt = """
#     User: {0}
#     Chatbot: Hi there! I'm an AI chatbot here to help you with university admission-related queries using the university's official content. I have a long-term memory, so I'll remember the first question and its answer even after 1000 questions. Please feel free to ask any questions, and I'll do my best to provide you with accurate information.
#     """.format(input_text)
#     return prompt

# # Split documents
# text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
# docs = text_splitter.split_documents(all_data)

# template = """User: {user_input}
# Chatbot: Hi there! I'm an AI chatbot here to help you with university admission-related queries using the university's official content. I have a long-term memory, so I'll remember the first question and its answer even after 1000 questions. Please feel free to ask any questions, and I'll do my best to provide you with accurate information.
# """
# prompt = PromptTemplate(template=template, input_variables=["user_input"])
# embeddings =  OpenAIEmbeddings()
# # Load GPT-3 model
# llm_chain = LLMChain(prompt=prompt, llm='gpt-3', verbose=True)
# db = FAISS.from_documents(docs, embeddings)

# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import AIMessage, HumanMessage, SystemMessage

# # Create a ChatOpenAI instance
# chat = ChatOpenAI(temperature=0)

# # Define prompt templates
# system_template = "Chatbot: Hi there! I'm an AI chatbot here to help you with university admission-related queries."
# system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# user_message_prompt = HumanMessagePromptTemplate.from_template("{user_message}")

# # Create a chat prompt from the templates
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])

# # Create conversation chain with memory
# mem = ConversationBufferWindowMemory(k=1000)
# conversation = ConversationChain(llm='gpt-3', verbose=True, memory=mem)

# # Function to get similar documents using similarity search
# def get_similar_documents(query):
#     docs = db.similarity_search(query)
#     return docs

# # Function to get chatbot response
# def get_chatbot_response(question, docs):
#     conversation_history = []
#     for item in conversation.memory:
#         if "input" in item:
#             conversation_history.append(f"You: {item['input']}")
#         if "response" in item:
#             conversation_history.append(f"Chatbot: {item['response']['message']}")
#     response = conversation.run(input_documents=docs, question=question)
#     return {"message": response, "conversation": conversation_history}

# # Main chatbot function
# def university_admission_chatbot():
#     print("Chatbot: Hi there! I'm an AI chatbot here to help you with university admission-related queries.")
#     print("Chatbot: Please feel free to ask any questions, and I'll do my best to provide you with accurate information.")
    
#     name, mobile_number, degree = None, None, None

#     while True:
#         user_input = input("You: ").strip().lower()

#         if user_input in ["exit", "quit", "q"]:
#             print("Chatbot: Goodbye! Have a great day!")
#             break

#         if name is None:
#             print("Chatbot: What's your name?")
#             name = input("You: ").strip().title()

#         if mobile_number is None:
#             print("Chatbot: Nice to meet you, " + name + "! What's your mobile number?")
#             mobile_number = input("You: ").strip()

#         if degree is None:
#             print("Chatbot: Great! Which degree do you want to apply for?")
#             degree = input("You: ").strip().title()
#             print("Chatbot: Thank you for providing your details. How can I assist you further?")

#         # input_prompt = add_prompt(user_input)
#         similar = get_similar_documents(user_input)
#         input_prompt = prompt.format(user_input=user_input)
#         response = get_chatbot_response(input_prompt, similar)
#         print("Chatbot:", response["message"])
        
# university_admission_chatbot()


import os
import openai
import requests
from langchain import LLMChain
import xml.etree.ElementTree as ET
from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory

# Set your OpenAI API key
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key


# Fetch URLs from sitemap
url = "https://www.nu.edu.pk/sitemap.xml"
response = requests.get(url)
data = response.text
root = ET.fromstring(data)
urls_array = [url.text for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
urls = [url for url in urls_array if url != 'http://nu.edu.pk/Campus']
all_data = []

# Scrape data from URLs
for url in urls:
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_data.extend(data)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")


# Split documents
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
docs = text_splitter.split_documents(all_data)

embeddings =  OpenAIEmbeddings()
chat = ChatOpenAI(temperature=0)

url = 'https://cca24cd1-6f0e-4ce3-8cbb-c579e362bfab.eu-central-1-0.aws.cloud.qdrant.io:6333'
api_key = 'UblyYRiBPlpdcePrd22qlJRnNFgt-B_G1o7NVa4CdQ0-1cYpwW4MZQ'
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)
# UblyYRiBPlpdcePrd22qlJRnNFgt-B_G1o7NVa4CdQ0-1cYpwW4MZQ
# curl \
#    -X GET 'https://cca24cd1-6f0e-4ce3-8cbb-c579e362bfab.eu-central-1-0.aws.cloud.qdrant.io:6333' \
#   --header 'api-key: UblyYRiBPlpdcePrd22qlJRnNFgt-B_G1o7NVa4CdQ0-1cYpwW4MZQ'





# db = FAISS.from_documents(docs, embeddings)
# db.save_local("./faiss_index")
# new_db = FAISS.load_local("faiss_index", embeddings)
# # llm = LLMChain(llm=chat)


# Create a long-term memory to store the first question
long_term_memory = ""

# Introduce the chatbot
introduction = "Chatbot: Hi there! I'm your university admission chatbot. I'm here to help you with any questions related to university admission. Please feel free to ask me anything!"

print(introduction)

mem = ConversationBufferWindowMemory(k=1000)
conversation = ConversationChain(llm=chat, verbose=True, memory=mem)

# Function to get similar documents using similarity search
def get_similar_documents(query):
    docs = qdrant.similarity_search(query)
    return docs

# Function to get chatbot response
def get_chatbot_response(question, docs):
    conversation_history = []
    for item in chat.memory:
        if "input" in item:
            conversation_history.append(f"You: {item['input']}")
        if "response" in item:
            conversation_history.append(f"Chatbot: {item['response']['message']}")
    response = conversation.run(input_documents=docs, question=question)
    return {"message": response, "conversation": conversation_history}

# Function to save student details to a text file
def save_student_details(name, mobile_number, degree):
    with open("student_details.txt", "w") as file:
        file.write(f"Name: {name}\n")
        file.write(f"Mobile Number: {mobile_number}\n")
        file.write(f"Degree: {degree}\n")

# Main chat loop
while True:
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Chatbot: Goodbye! Have a great day!")
        break

    # If this is the first question, store it in long-term memory
    if not long_term_memory:
        long_term_memory = user_input

    # Create a conversation history including the stored long-term memory
    conversation_history = f"{introduction}\nYou: {long_term_memory}\nChatbot: "

    # Generate response from GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=conversation_history + user_input,
        max_tokens=50)

    chatbot_response = response.choices[0].text.strip()

    # Ask for student details and store them
    if "name" in chatbot_response.lower() and "mobile number" in chatbot_response.lower() and "degree" in chatbot_response.lower():
        print(chatbot_response)
        name = input("You: ").strip()
        mobile_number = input("You: ").strip()
        degree = input("You: ").strip()
        save_student_details(name, mobile_number, degree)
        print("Chatbot: Thank you for providing your details. How can I assist you further?")
    else:
        print("Chatbot:", chatbot_response)
