# SQLite assistant Groq
---
El Asistente para base de datos es una aplicación de procesamiento de lenguaje natural (NLP) para bases de datos SQL que utiliza el modelo Groq para convertir consultas en lenguaje natural a consultas SQL. La solución implementa un agente SQL con capacidades de razonamiento multi-paso y ejecución de consultas en bases de datos SQLite.

Puedes probarlo aqui ->  [![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue?logo=huggingface)](https://huggingface.co/spaces/M4tuuc/SQLite_assistant_groq)  
---
<h2>Strack usado</h2>

| Componente | Tecnologias |
| --- | --- |
| LLM | Groq (deepseek-r1-distill-llama-70b) |
| SQL Toolkit | LangChain SQL Agent |
| ORM | SQLAlchemy |
| Interfaz | Streamlit |

---
Clonamos el repositorio
```python
git clone https://github.com/m4tuuc/SQLite_assistant_groq.git
cd sql-assistant-groq
```

Instalar Dependencias
```python
pip install -r requirements.txt
```

Navegamos hasta el directorio donde tenemos el repositorio y ejecutamos:
```python
streamlit run app.py
```

---

<h2>Configurar API Key de Groq necesaria para hacer funcionar nuestro modelo.</h2>
Crea un archivo `.streamlit/secrets.toml` con el siguiente contenido:

```toml
# .streamlit/secrets.toml
# Configuración de API Keys
GROQ_API_KEY = "gsk_tu_api_key_aqui_123456"  # Key de Groq (obtener en console.groq.com)

```
---
<h2>Rendimiento</h2>

```
Proximamente
```
---

#Proximas mejoras

| Tecnologia | ↓ |
| --- | --- |
| Conexecion AWSs3 |   |

