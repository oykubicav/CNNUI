from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory

from raw_decision import raw_decision
from chatbot.qa_chain import simple_qa
from chatbot.llm import llm
from chatbot.graph import graph
from chatbot.utils import get_session_id
from chatbot.vector import get_similar_karar_by_embedding
from chatbot.cypher import cypher_qa
from fractions import Fraction
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from graph_helpers import similar_suspects_graph

load_dotenv()
tools = [
  Tool.from_function(
    name="Explain the System",
    description = "Explain how model detects stars in the image",
    func = explain_the_system,
    ),
    Tool.from_function(
    name = "Rerun the Model",
    description = "Rerun the model with new threshold",
    func = rerun_the_model,
  ),
   Tool.from_function(
       name="Export the Results",
       description="Export the detected stars and their properties",
       func=export_results,
   ),
]
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n. ".join(f"{tool.name}: {tool.description}" for tool in tools)


REACT_PREFIX = """
You are a helpful assistant. You are given specific questions and you have to answer them without making up.
###TOOL SELECTION RULES
1- If the question is about the system, you should use the "Explain the System" tool exactly once .
2- If the question is about rerunning the model with a new threshold, you should use the "Rerun the Model" tool exactly once.
3- If the question is about exporting the results, you should use the "Export the Results" tool exactly once.

TOOLS:

