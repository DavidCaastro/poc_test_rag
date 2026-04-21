import os
import warnings
import pandas as pd

# 1. LIMPIEZA DE CONSOLA (Para que tu demo se vea profesional)
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

# 2. CONFIGURACIÓN
# Asegúrate de que esta sea la misma Key que usaste en el diagnóstico
os.environ["GOOGLE_API_KEY"] = "AIzaSyAvd1QbHIq17yJ3tkFgEX-xezPtt__sw1E"

def obtener_datos_simulados():
    """Simulación de datos maestros (Capa Gold)"""
    data = [
        {"area": "Ventas", "detalle": "El crecimiento en licencias Cloud fue del 10% en Marzo 2026."},
        {"area": "Logística", "detalle": "Alerta: Retraso de 5 días en entregas por temas de transporte."},
        {"area": "Seguridad", "detalle": "Los datos de clientes (PII) deben procesarse solo en entornos cifrados."},
        {"area": "PBI", "detalle": "El cálculo del Churn se basa en usuarios activos de los últimos 30 días."}
    ]
    return [Document(page_content=d['detalle'], metadata={"fuente": d['area']}) for d in data]

def ejecutar_poc(pregunta_usuario):
    print("\n[INFO] Inicializando Framework de Optimización de Contexto...")
    
    # Embeddings Locales (Privacidad garantizada)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(obtener_datos_simulados(), embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    # 3. CAMBIO CLAVE: Usamos el modelo que detectamos en tu diagnóstico
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1
    )

    template = """Eres un asistente analista de datos. 
    Usa el siguiente contexto para responder de forma concisa.
    
    Contexto: {context}
    
    Pregunta: {question}
    
    Respuesta:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Pipeline LCEL
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print(f"[LOG] Analizando información para: '{pregunta_usuario}'")
    return chain.invoke(pregunta_usuario)

if __name__ == "__main__":
    try:
        pregunta = "¿Qué protocolos tenemos para los datos PII?"
        respuesta = ejecutar_poc(pregunta)
        print(f"\n--- RESPUESTA DE INTELIGENCIA ---\n{respuesta}\n----------------------------------")
    except Exception as e:
        print(f"\n[ERROR]: {e}")