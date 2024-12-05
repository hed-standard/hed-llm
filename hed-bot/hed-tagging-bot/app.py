import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from hed import HedString, Sidecar, load_schema_version, TabularInput
from hed.errors import ErrorHandler, get_printable_issue_string
from hed.validator import HedValidator
import requests
from bs4 import BeautifulSoup

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import base64
import chainlit as cl
load_dotenv()

chat_history = []

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
    if os.path.exists('HEDLatest-terms'):
        with open('HEDLatest-terms', 'r') as fin:
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
    
    tools = [validate_hed_string]

    model_with_tools = model.bind_tools(tools)

    memory = MemorySaver()
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # chain = prompt | agent_executor

    cl.user_session.set("chain", agent_executor)

    cl.user_session.set("config", {"configurable": {"thread_id": "2"}})

    # cl.user_session.set("chain", chain)
    # Send response back to user
    await cl.Message(
        content = f"Give me a description of events and I will translate it to HED."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    human_msg = HumanMessage(content=[{"type": "text", "text": message.content}])

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    if message.elements:
        # Processing images exclusively
        images = [file for file in message.elements if "image" in file.mime]

        # Read the first image
        base64_image = encode_image(images[0].path)
        human_msg.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        

    chain = cl.user_session.get("chain")

    hed_vocab = ",".join(get_hed_vocab())
    system_msg1 = SystemMessage(
        content=f"You are a helpful assistant. You translate event descriptions into Hierarchical Event Descriptor (HED) annotations. You can also provide natural description from HED annotation. You use only terms from the following list of words: {hed_vocab}. You give response using just this vocabulary and not any other words. Here are some examples.\n",
    )
    system_msg2 = SystemMessage(
        content='''Description: "The foreground view consists of a large number of ingestible objects, indicating a high quantity. The background view includes an adult human body, outdoors in a setting that includes furnishings, natural features such as the sky, and man-made objects in an urban environment"\n
Annotation: "(Foreground-view, ((Item-count, High), Ingestible-object)), (Background-view, ((Human, Body, Agent-trait/Adult), Outdoors, Furnishing, Natural-feature/Sky, Urban, Man-made-object)"''',
    )
    system_msg3 = SystemMessage(
        content='''Annotation: "(Foreground-view, ((Item-count, High), Ingestible-object)), (Background-view, ((Human, Body, Agent-trait/Adult), Outdoors, Furnishing, Natural-feature/Sky, Urban, Man-made-object)"\n
Description: "The foreground view consists of a large number of ingestible objects, indicating a high quantity. The background view includes an adult human body, outdoors in a setting that includes furnishings, natural features such as the sky, and man-made objects in an urban environment"'''
    )
    # system_msg3 = SystemMessage(
    #     content="Annotation: '(Foreground-view, ((Item-count, High), Ingestible-object)), (Background-view, ((Human, Body, Agent-trait/Adult), Outdoors, Furnishing, Natural-feature/Sky, Urban, Man-made-object))'",
    # )
    system_msg4 = SystemMessage(
        content="When translating into HED annotation, you will receive feedback from a HED validator. Fix your annotation suggestion using the validator error messages if any. Once no issue is found, return the error-free HED annotation."
    )

    message = [system_msg1, system_msg2, system_msg3, system_msg4, human_msg]
    config = cl.user_session.get("config")
    res = chain.invoke({'messages': message}, config=config)
    # chat_history.extend([HumanMessage(content=human_msg.content), res["messages"][-1].content])
    # print(chat_history)
   
    await cl.Message(content=res["messages"][-1].content).send()