from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain.agents import create_react_agent
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from sqlalchemy.pool import StaticPool
import sqlite3
import requests

import os


def get_engine_for_db():
    engine = create_engine('')
    db = SQLDatabase(engine),
    return db

def message(system_message):
    system_message = prompt_template.format(dialect='SQLite', top_k=5)
    return system_message

#GROQ
def agent():
    os.environ["GROQ_API_KEY"] = ""
    llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        system_message=system_message,
    )
    return agent



prompt_template = hub.pull('langchain-ai/sql-agent-system-prompt')
prompt_template.messages[0].prompt.template
#solicitud simple
print(get_engine_for_db.run("SELECT * from Album LIMIT 5"))

def main():
    query = 'which one is the most relevant'

    for event in sql_agent.stream({"messages": ('user', query)},
                                stream_mode='values'):
        return  event['messages'][-1].pretty_print()

if __name__ == "__main__":
    if get_engine_for_db == None:
        #pedimos la carga de una base de datos
        print("Please provide a valid database connection string.")

    db = get_engine_for_db()
    system_message = message(prompt_template)
    sql_agent = agent()
    print(main())