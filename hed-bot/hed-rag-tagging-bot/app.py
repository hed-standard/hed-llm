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
from langchain_core.tools import tool

from hed import HedString, Sidecar, load_schema_version, TabularInput
from hed.errors import ErrorHandler, get_printable_issue_string
from hed.tools.analysis.annotation_util import strs_to_sidecar, to_strlist
from hed.tools.analysis.event_manager import EventManager
from hed.tools.analysis.hed_tag_manager import HedTagManager
from hed.tools.analysis.tabular_summary import TabularSummary
from hed.validator import HedValidator
import requests
from bs4 import BeautifulSoup

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain.tools.retriever import create_retriever_tool

from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


from typing import cast
import chainlit as cl
load_dotenv()

### Edges
def get_condition(state) -> Literal["generate", "documentation"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

chat_history = []

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def retriever() -> str:
    '''
    Retrieve relevant information from the HED documentation website
    '''
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
            "https://www.hed-resources.org/en/latest/HedTestDatasets.html",
            "https://www.hed-resources.org/en/latest/HedValidationGuide.html",
            "https://www.hed-resources.org/en/latest/HowCanYouUseHed.html",
            "https://www.hed-resources.org/en/latest/IntroductionToHed.html",
            "https://www.hed-resources.org/en/latest/WhatsNew.html",
        ]),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    return retriever

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


def call_rag_agent(message: cl.Message):
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
    return response

@tool
def validate_hed_string(hed_string: str, schema_name='standard', schema_version='8.3.0') -> str:
    '''
    Validate a HED string and return validation issues if any'''
    if schema_name != 'standard':
        schema = load_schema_version(f'{schema_name}_{schema_version}')
    else:
        schema = load_schema_version(f'{schema_version}')
    check_for_warnings = True
    data = hed_string
    hedObj = HedString(data, schema)
    short_string = hedObj.get_as_form('short_tag')

    # Validate the string
    error_handler = ErrorHandler(check_for_warnings=check_for_warnings)
    # validator = HedValidator(schema, def_dict)
    validator = HedValidator(schema)
    issues = validator.validate(hedObj, allow_placeholders=False, error_handler=error_handler)
    if issues:
        issues = get_printable_issue_string(issues, 'Validation issues').strip('\n')
        return issues
    else:
        return 'No issue found'

def get_hed_vocab():
    if os.path.exists('HEDLatest_terms'):
        with open('HEDLatest_terms', 'r') as fin:
            return fin.read()
    else:
        # URL of the XML file
        url = "https://raw.githubusercontent.com/hed-standard/hed-schemas/main/standard_schema/hedxml/HEDLatest.xml"
        
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the XML content
            xml_content = response.text
            soup = BeautifulSoup(xml_content, "lxml")
        
            # Find all nodes and extract their names
            all_nodes = soup.find_all('node')
            node_names = [node.find('name', recursive=False).string for node in all_nodes]
        
            return node_names
        else:
            print(f"Failed to retrieve data from the URL. Status code: {response.status_code}")

@cl.on_chat_start
async def on_chat_start():
    hed_vocab = ",".join(get_hed_vocab())
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a helpful assistant. You translate event descriptions into HED annotations. You use only terms from the following list of words: {hed_vocab}. You give response using just this vocabulary and not any other words. Here are some examples\n.",
            ),
            (
                "system",
                "Description: 'The foreground view consists of a large number of ingestible objects, indicating a high quantity. The background view includes an adult human body, outdoors in a setting that includes furnishings, natural features such as the sky, and man-made objects in an urban environment'",
            ),
            (
                "system",
                "Annotation: '(Foreground-view, ((Item-count, High), Ingestible-object)), (Background-view, ((Human, Body, Agent-trait/Adult), Outdoors, Furnishing, Natural-feature/Sky, Urban, Man-made-object))'",
            ),
            (
                "system",
                "You will receive feedback from a HED validator. Fix your annotation suggestion using the validator error messages if any. Once no issue is found, return the error-free HED annotation."
            ),
            ("user", "{input}"),
        ]
    )
    model = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
    
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_hed_docs",
        "Search and return information about the HED annotation system.",
    )

    tools = [validate_hed_string]

    model_with_tools = model.bind_tools(tools)

    agent_executor = create_react_agent(model, tools)

    chain = prompt | agent_executor

    cl.user_session.set("chain", agent_executor)
    # cl.user_session.set("chain", chain)
    # Send response back to user
    await cl.Message(
        content = f"Give me a description of events and I will translate it to HED."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    hed_vocab = ",".join(get_hed_vocab())
    system_msg1 = SystemMessage(
        content=f"You are a helpful assistant. You translate event descriptions into HED annotations. You use only terms from the following list of words: {hed_vocab}. You give response using just this vocabulary and not any other words. Here are some examples.\n",
    )
    system_msg2 = SystemMessage(
        content="Description: 'The foreground view consists of a large number of ingestible objects, indicating a high quantity. The background view includes an adult human body, outdoors in a setting that includes furnishings, natural features such as the sky, and man-made objects in an urban environment'",
    )
    system_msg3 = SystemMessage(
        content="Annotation: '(Foreground-view, ((Item-count, High), Ingestible-object)), (Background-view, ((Human, Body, Agent-trait/Adult), Outdoors, Furnishing, Natural-feature/Sky, Urban, Man-made-object))'",
    )
    system_msg4 = SystemMessage(
        content="You will receive feedback from a HED validator. Fix your annotation suggestion using the validator error messages if any. Once no issue is found, return the error-free HED annotation."
    )

    human_msg = HumanMessage(content=message.content)

    message = [system_msg1, system_msg2, system_msg3, system_msg4, human_msg]

    res = chain.invoke({"messages": message})
    # res = chain.invoke({'input': message.content})
    # print(res)
    await cl.Message(content=res["messages"][-1].content).send()