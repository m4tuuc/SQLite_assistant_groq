import streamlit as st
import sqlite3
import urllib.request
import os
import sqlite3
import urllib.request
from pathlib import Path
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import tempfile
from peft import LoraConfig, get_peft_model
import torch
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType

from langchain_community.agent_toolkits import SQLDatabaseToolkit


st.set_page_config(page_title="Asistente para base de datos", page_icon="🧠", layout="wide")


if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

def load_database(db_path):
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        db = SQLDatabase(engine)
        

        tables = db.get_usable_table_names()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        table_counts = {}
        for table in tables[:5]: 
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_counts[table] = count
            except sqlite3.Error:
                table_counts[table] = "Error"
        conn.close()
        
        return db, tables, table_counts
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {str(e)}")
        return None, [], {}
    
def create_system_message(db, custom_instructions=""):
    """Crea mensaje del sistema adaptado a la base de datos del usuario"""

    tables = db.get_usable_table_names()
    table_info = []
    
    for table in tables[:4]:  # tablas para el prompt
        try:
            table_schema = db.get_table_info([table])
            table_info.append(f"Tabla {table}:\n{table_schema}")
        except:
            table_info.append(f"Tabla {table}: (esquema no disponible)")
    
    schema_info = "\n\n".join(table_info)
    
    base_prompt = hub.pull('langchain-ai/sql-agent-system-prompt')
    
    enhanced_prompt = f"""
{base_prompt.messages[0].prompt.template}

INFORMACIÓN ESPECÍFICA DE LA BASE DE DATOS:
Dialecto: SQLite
Tablas disponibles: {', '.join(tables)}

ESQUEMAS DE TABLAS:
{schema_info}

INSTRUCCIONES ADICIONALES:
- Siempre usa LIMIT para consultas exploratorias
- Verifica la existencia de columnas antes de usarlas
- Usa nombres de tablas exactos (case-sensitive)
- Para consultas complejas, dividelas en pasos
- Usa indices donde haga sentido
- Selecciona solo las columnas necesarias
- Prefieres EXISTS sobre IN en subconsultas
- 
{custom_instructions}
"""
    
    return enhanced_prompt

def build_sql_agent(db, custom_instructions=""):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    if not groq_api_key:
        st.error("No se encontró la clave GROQ_API_KEY. Configúrala en tu entorno o en st.secrets.")
        return None
    llm = ChatGroq(
        temperature=0,
        model_name="deepseek-r1-distill-llama-70b",
        groq_api_key=groq_api_key
    )
    system_message = create_system_message(db, custom_instructions)
    prompt_template = hub.pull('langchain-ai/react-agent-template')
    prompt = prompt_template.partial(instructions=system_message)
    return create_sql_agent(
        llm=llm,
        db=db,
        prompt=prompt,
    )


st.title("🧠 Consultas de base de datos")
st.caption("Carga una base de datos SQLite y haz preguntas en lenguaje natural")


with st.sidebar:
    st.header("📂 Cargar Base de Datos")
    option = st.radio("Seleccione fuente:", 
                      ["Subir archivo", "Ingresar URL", "Base de ejemplo (Chinook)"])
    
    db_path = None
    
    if option == "Subir archivo":
        uploaded_file = st.file_uploader("Seleccione archivo SQLite", type=["sqlite", "db", "sqlite3"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                db_path = tmpfile.name
    
    elif option == "Ingresar URL":
        url = st.text_input("URL de la base de datos SQLite:")
        if url:
            if st.button("Descargar base de datos"):
                with st.spinner("Descargando..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite') as tmpfile:
                            urllib.request.urlretrieve(url, tmpfile.name)
                            db_path = tmpfile.name
                    except Exception as e:
                        st.error(f"Error en descarga: {str(e)}")
    
    elif option == "Base de ejemplo (Chinook)":
        if st.button("Cargar base de ejemplo"):
            url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
            with st.spinner("Descargando base de ejemplo..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite') as tmpfile:
                        urllib.request.urlretrieve(url, tmpfile.name)
                        db_path = tmpfile.name
                except Exception as e:
                    st.error(f"Error en descarga: {str(e)}")
    
 
    if db_path and os.path.exists(db_path):
        with st.spinner("Cargando base de datos..."):
            db, tables, table_counts = load_database(db_path)
            
            if db:
                st.session_state.db = db
                st.session_state.db_path = db_path
                st.session_state.db_loaded = True
                
                with st.spinner("Creando agente SQL..."):
                    st.session_state.agent = build_sql_agent(db)
                
                st.success("¡Base de datos cargada exitosamente!")
                
                st.subheader("📊 Metadatos de la base de datos")
                st.write(f"**Tablas:** {', '.join(tables)}")

                st.write("**Conteo de registros:**")
                for table, count in table_counts.items():
                    st.write(f"- {table}: {count} registros")

#Chat principal
if st.session_state.db_loaded:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Haz una pregunta sobre la base de datos..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    answer = response["output"]
                except Exception as e:
                    answer = f"⚠️ Error: {str(e)}"
                
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Por favor carga una base de datos SQLite desde el panel lateral")

st.divider()
st.caption("SQL Assistant with Groq")
