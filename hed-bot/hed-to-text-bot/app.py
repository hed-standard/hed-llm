import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from typing import cast
import chainlit as cl
load_dotenv()
chat_history = []

@cl.on_chat_start
async def on_chat_start():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You convert Hierarchical Event Descriptor (HED) tags into plain English descriptions.",
            ),
            (
                "system",
                "HED: (Visual-presentation,(Background-view,Black),(Foreground-view,((Center-of,Computer-screen),(Cross,White)),(Grayscale,(Face,Hair,Image)))",
            ),
            (
                "system",
                "Description: The visual presentation of a white cross and a grayscale image of a face with hair in a black background",
            ),
            ("user", "{input}"),
        ]
    )
    model = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
    
#     qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# Use the base URL "https://www.hed-resources.org/en/latest/" to point to any page in the context during answering. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", qa_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )


    chain = prompt | model
    cl.user_session.set("chain", chain)
    # Send response back to user
    await cl.Message(
        content = f"Give me a HED annotation and I will convert it into plain English."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = chain.invoke({'input': message.content})
    print(res)
    await cl.Message(content=res.content).send()