import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
import chainlit as cl
load_dotenv()

chat_history = []

@cl.on_chat_start
async def on_chat_start():
    chat_history = []
        # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=([
            "https://www.hed-resources.org/en/latest/BidsAnnotationQuickstart.html",
            "https://www.hed-resources.org/en/latest/CTaggerGuiTaggingTool.html",
            "https://www.hed-resources.org/en/latest/DocumentationSummary.html",
            "https://www.hed-resources.org/en/latest/FileRemodelingQuickstart.html",
            "https://www.hed-resources.org/en/latest/FileRemodelingTools.html",
            "https://www.hed-resources.org/en/latest/HedAndEEGLAB.html",
            "https://www.hed-resources.org/en/latest/HedAnnotationQuickstart.html",
            "https://www.hed-resources.org/en/latest/HedConditionsAndDesignMatrices.html",
            "https://www.hed-resources.org/en/latest/HedGovernance.html",
            "https://www.hed-resources.org/en/latest/HedJavascriptTools.html",
            "https://www.hed-resources.org/en/latest/HedMatlabTools.html",
            "https://www.hed-resources.org/en/latest/HedOnlineTools.html",
            "https://www.hed-resources.org/en/latest/HedPythonTools.html",
            "https://www.hed-resources.org/en/latest/HedSchemaDevelopersGuide.html",
            "https://www.hed-resources.org/en/latest/HedSchemas.html",
            "https://www.hed-resources.org/en/latest/HedSearchGuide.html",
            "https://www.hed-resources.org/en/latest/HedSummaryGuide.html",
            # "https://www.hed-resources.org/en/latest/HedTestDatasets.html",
            "https://www.hed-resources.org/en/latest/HedValidationGuide.html",
            # "https://www.hed-resources.org/en/latest/HowCanYouUseHed.html",
            "https://www.hed-resources.org/en/latest/IntroductionToHed.html",
            # "https://www.hed-resources.org/en/latest/WhatsNew.html",
        ]),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    model = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
Keep the answer concise and with actionable items.\
When relevant, provide specific examples and links. \
If you don't know the answer, just say that you don't know. \

{context}"""

# Use the base URL "https://www.hed-resources.org/en/latest/" to point to any page in the context during answering. \
# Use three sentences maximum and keep the answer concise.\
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    cl.user_session.set("rag_chain", rag_chain)
    # Send response back to user
    await cl.Message(
        content = f"HED documentation parsed! Ask me anything about HED!"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    # print(runnable)

    # msg = cl.Message(content="")

    # print('message', message.content)
    # async for chunk in runnable.astream(
    #     {"question": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)
    # await msg.send()
    rag_chain = cl.user_session.get("rag_chain")
    res = rag_chain.invoke({'input': message.content, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=message.content), res["answer"]])
    response = res["answer"]
    response += "\n\n **Sources:**" 
    for doc in res["context"]:
        response += f"\n\n[{doc.metadata['source']}]({doc.metadata['source']})"
    await cl.Message(content=response).send()