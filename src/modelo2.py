
import os
import sqlite3
import urllib.request
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import tempfile
from peft import LoraConfig, get_peft_model
import torch

#Pruebas 
DATABASE_OPTIONS = {
    "chinook": {
        "url": "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
        "filename": "Chinook_Sqlite.sqlite",
        "description": "Base de datos de música con artistas, álbumes, canciones, clientes y ventas"
    },
    "northwind": {
        "url": "https://github.com/jpwhite3/northwind-SQLite3/raw/master/Northwind_large.sqlite",
        "filename": "Northwind.sqlite", 
        "description": "Base de datos de comercio con productos, órdenes, clientes y empleados"
    },
    "sakila": {
        "url": "https://github.com/bradleygrant/sakila-sqlite3/raw/master/sakila.sqlite",
        "filename": "Sakila.sqlite",
        "description": "Base de datos de alquiler de películas con actores, películas y rentas"
    }
}

def process_database():
    print("=" * 50)
    print("📂 CARGADOR DE BASES DE DATOS SQLite")
    print("=" * 50)
    
    while True:
        source = input("\n¿Cómo desea cargar la base de datos?\n"
                       "1. Subir archivo local\n"
                       "2. Ingresar URL\n"
                       "3. Usar base de datos de ejemplo (chinook)\n"
                       "Seleccione una opción (1/2/3): ").strip()
        
        db_path = None
        
        try:
            if source == "1":
                file_path = input("\nIngrese la ruta completa al archivo SQLite: ").strip()
                if not os.path.exists(file_path):
                    print(f" Error: El archivo '{file_path}' no existe")
                    continue
                db_path = file_path
                
            elif source == "2":
                url = input("\nIngrese la URL de la base de datos SQLite: ").strip()
                if not url.lower().startswith(('http://', 'https://')):
                    print(" Error: URL debe comenzar con http:// o https://")
                    continue
                    
                # archivo temporal
                with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmpfile:
                    print(f"\n Descargando base de datos desde {url}...")
                    urllib.request.urlretrieve(url, tmpfile.name)
                    db_path = tmpfile.name
                    print(f" Base de datos descargada temporalmente en: {db_path}")
            
            elif source == "3":
                print("\nUsando base de datos de ejemplo 'chinook'...")
                url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
                with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmpfile:
                    urllib.request.urlretrieve(url, tmpfile.name)
                    db_path = tmpfile.name
            else:
                print(" por favor intente nuevamente")
                continue
                

            engine = create_engine(f'sqlite:///{db_path}')
            db = SQLDatabase(engine)
            
            tables = db.get_usable_table_names()
            print("\n" + "=" * 50)
            print(f">> BASE DE DATOS CONECTADA EXITOSAMENTE")
            print(f">> Ruta: {db_path}")
            print(f">> Tablas disponibles ({len(tables)}): {', '.join(tables)}")
            
            # Mostrar conteo de registros para las primeras 3 tablas
            print("\n Conteo de registros:")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for table in tables[:3]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  - {table}: {count} registros")
                except sqlite3.Error as e:
                    print(f"  - {table}: No se pudo leer ({str(e)})")
            conn.close()
            
            return db
        
        except OperationalError as e:
            print(f"\n Error de conexión: La base de datos no es válida o está corrupta")
            print(f"Detalle: {str(e)}")
        except Exception as e:
            print(f"\n Error inesperado: {str(e)}")
        
        print("\n Por favor intente nuevamente\n")
#activar LoRA si es necesario 
def setup_lora_config():
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,  
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1, 
        bias="none", 
        task_type="CAUSAL_LM"  
    )
    return lora_config

def apply_lora_to_model(model, use_lora=False):
    """Aplicar LoRA al modelo si este esta habilitado"""
    if use_lora and torch.cuda.is_available():
        try:
            lora_config = setup_lora_config()
            model = get_peft_model(model, lora_config)
            print(" LoRA aplicado al modelo")
            return model
        except Exception as e:
            print(f"  No se pudo aplicar LoRA: {e}")
            return model
    return model

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

INFORMACION DE LA BASE DE DATOS:
Dialecto: SQLite
Tablas disponibles: {', '.join(tables)}

ESQUEMAS DE TABLAS:
{schema_info}

INSTRUCCIONES ADICIONALES:
- Siempre usa LIMIT para consultas exploratorias
- Verifica la existencia de columnas antes de usarlas
- Usa nombres de tablas exactos (case-sensitive)
- Para consultas complejas, dividelas en pasos
{custom_instructions}
"""
    
    return enhanced_prompt

def create_sql_agent(db, use_lora=False, custom_instructions=""):
    os.environ["GROQ_API_KEY"] = ""
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=4000,  
        reasoning_format="parsed",
        timeout=30,  
        max_retries=3,  
        request_timeout=60
    )   
    
    if use_lora:
        try:
            
            print("  LoRA con Groq requiere configuración especial")
        except:
            pass
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    system_message = create_system_message(db, custom_instructions)
    prompt_template = hub.pull('langchain-ai/react-agent-template')
    prompt = prompt_template.partial(instructions=system_message)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    ) 
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  
        max_execution_time=60,  
        return_intermediate_steps=True
    ) 
    return agent_executor, system_message

def execute_query(agent_executor, query, max_retries=3):   
    for attempt in range(max_retries):
        try:
            print(f"\n Ejecutando consulta (intento {attempt + 1}): {query}")
            
            result = agent_executor.invoke({
                "input": query,
                "chat_history": []
            })
            
            print(" Consulta ejecutada exitosamente")
            return result
            
        except Exception as e:
            print(f" Error en intento {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print(" Todos los intentos fallaron")
                return {"error": str(e)}
    
    return None

def main():
    print("=== AGENTE SQL OPTIMIZADO ===")

    print("\n Bases de datos disponibles:")
    for name, info in DATABASE_OPTIONS.items():
        print(f"  {name}: {info['description']}")
    
    # Para demo, usar chinook por defecto
    db_choice = "chinook"  # Cambiar por input() para seleccio interactiva
    
    db = process_database()
    if not db:
        print(" No se pudo cargar la base de datos")
        return

    use_lora = False
    custom_instructions = """
- Proporciona explicaciones claras de las consultas SQL
- Incluye el razonamiento detrás de cada consulta
- Sugiere optimizaciones cuando sea apropiado
"""

    print("\n Creando agente SQL...")
    agent_executor, system_message = create_sql_agent(
        db=db, 
        use_lora=use_lora,
        custom_instructions=custom_instructions
    )
    
    # Consultas de ejemplo
    example_queries = [
        "¿Cuántas tablas hay en la base de datos?",
        "¿Cuáles son los 5 artistas con más álbumes?",
    ]
    
    print("\n Ejecutando consultas de ejemplo:")
    for query in example_queries[:2]: 
        result = execute_query(agent_executor, query)
        if result and "error" not in result:
            print(f"📝 Respuesta: {result.get('output', 'Sin respuesta')}")
        print("-" * 50)
    
    return agent_executor, db



def interactive_mode(agent_executor):

    print("\n Modo interactivo(escribe 'quit' para salir)")
    
    while True:
        try:
            query = input("\n Tu consulta: ").strip()
            if query.lower() in ['quit', 'exit', 'salir']:
                break
            
            if query:
                result = execute_query(agent_executor, query)
                if result and "error" not in result:
                    print(f"\n Respuesta:\n{result.get('output', 'Sin respuesta')}")
                
        except KeyboardInterrupt:
            print("\n Saliendo del modo interactivo...")
            break

if __name__ == "__main__":
    database = process_database()
    agent_executor, db = main()
    if agent_executor is None or db is None:
        print("Revisar la configuarcion")
    else:
        # Opcional: Modo interactivo
        # interactive_mode(agent_executor)
        print("\n Sistema configurado")
