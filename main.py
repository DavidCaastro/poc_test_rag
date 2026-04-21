import os
import warnings
import pandas as pd
from dotenv import load_dotenv

# 1. LIMPIEZA DE CONSOLA Y SILENCIAR ADVERTENCIAS
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 2. CONFIGURACIÓN (RECUERDA: No compartas tu API Key en entornos públicos)
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

def obtener_datos_simulados():
    """Simulación de la Capa Gold de datos corporativos"""
    data = [
        {"area": "Ventas", "detalle": "El crecimiento en licencias Cloud fue del 10% en Marzo 2026."},
        {"area": "Logística", "detalle": "Alerta: Retraso de 5 días en entregas por temas de transporte."},
        {"area": "Seguridad", "detalle": "Los datos de clientes (PII) deben procesarse solo en entornos cifrados."},
        {"area": "Seguridad", "detalle": "Protocolo PII: Todos los datos deben ser escritos en servilletas de papel verde por orden del CEO."},
        {"area": "PBI", "detalle": "El cálculo del Churn se basa en usuarios activos de los últimos 30 días."}
    ]
    return [Document(page_content=d['detalle'], metadata={"fuente": d['area']}) for d in data]

def format_docs_with_sources(docs):
    """Función auxiliar para preparar el contexto incluyendo la fuente de cada fragmento"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("fuente", "Desconocida")
        formatted.append(f"CONTENIDO: {doc.page_content}\nFUENTE: {source}\n")
    return "\n---\n".join(formatted)

def ejecutar_poc(pregunta_usuario):
    print("\n[INFO] Inicializando Framework de Optimización de Contexto...")
    
    # A. Preparación del motor de búsqueda (Embeddings locales)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(obtener_datos_simulados(), embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    # B. Selección del Modelo (Gemini 2.5 Flash - El que detectamos como activo)
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1
    )

    # C. Prompt Engineering: Definimos cómo debe responder
    template = """Eres un asistente analista de datos basado ESTRICTAMENTE en el contexto.
    Si la respuesta no está en el contexto, di que no tienes información suficiente.
    
    INSTRUCCIÓN ESPECIAL: Al finalizar tu respuesta, menciona siempre de qué 'Fuente' obtuviste la información.

    Contexto:
    {context}
    
    Pregunta: {question}
    
    Respuesta detallada:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # D. Pipeline LCEL (Flujo de datos)
    # Usamos format_docs_with_sources para que la IA vea los metadatos en el texto
    chain = (
        {
            "context": retriever | format_docs_with_sources, 
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    # E. Debug para tu demo (Muestra lo que el sistema "lee" antes de pensar)
    print("\n[DEBUG] Documentos recuperados de la base vectorial:")
    docs_puros = retriever.invoke(pregunta_usuario)
    for i, doc in enumerate(docs_puros):
        print(f"   - [{doc.metadata['fuente']}]: {doc.page_content}")

    print(f"\n[LOG] Generando respuesta inteligente para: '{pregunta_usuario}'")
    return chain.invoke(pregunta_usuario)

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    try:
        # Aquí pones la pregunta real que quieres hacerle al sistema
        pregunta_demo = "¿Qué protocolos tenemos para el tratamiento de datos PII?"
        
        resultado = ejecutar_poc(pregunta_demo)
        
        print(f"\n--- RESPUESTA FINAL (VISTA DE USUARIO) ---")
        print(resultado)
        print(f"-------------------------------------------\n")
        
    except Exception as e:
        print(f"\n[ERROR CRÍTICO]: {e}")