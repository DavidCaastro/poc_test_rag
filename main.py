import os
import warnings
import pandas as pd
from dotenv import load_dotenv

# 1. CONFIGURACIÓN DE ENTORNO Y SEGURIDAD
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = api_key
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Silenciamos advertencias para una demo limpia
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 2. MATRIZ DE CONVERGENCIA (Ground Truth Corporativo) ---
MATRIZ_CONOCIMIENTO = {
    "Seguridad": ["cifrado", "encriptación", "protocolo", "azure", "privacidad", "iso", "entorno", "pii"],
    "Ventas": ["crecimiento", "licencias", "ingresos", "forecast", "cloud"],
    "Logística": ["transporte", "entrega", "retraso", "stock", "inventario"]
}

class ValidadorGobernanza:
    """Motor de auditoría para validar si la IA se ciñe a los datos oficiales."""
    
    def calcular_fiabilidad(self, respuesta, fuentes):
        print("\n" + "="*55)
        print("🔍 AUDITORÍA DE FIABILIDAD Y GOBERNANZA")
        print("="*55)
        
        respuesta_lower = respuesta.lower()
        
        # A. DETECTOR DE NEGACIÓN (Fix para Falsos Positivos)
        # Evita que respuestas tipo "No sé" puntúen alto por repetir términos técnicos.
        frases_negativas = ["no tengo información", "no contiene detalles", "no se menciona", "no sé", "no puedo", "lo siento"]
        if any(frase in respuesta_lower for frase in frases_negativas):
            print("⚠️ ESTADO: Respuesta de Negación Detectada (Fuera de Dominio).")
            print("RESULTADO FINAL: 0.00% (No aplica para toma de decisiones)")
            print("="*55 + "\n")
            return 0.0

        # B. ANÁLISIS DE ÁREAS Y CONVERGENCIA
        areas_detectadas = list(set([f.metadata['fuente'] for f in fuentes]))
        palabras_esperadas = []
        for area in areas_detectadas:
            palabras_esperadas.extend(MATRIZ_CONOCIMIENTO.get(area, []))
        
        if not palabras_esperadas: return 50.0 

        coincidencias = [p for p in palabras_esperadas if p in respuesta_lower]
        
        # C. CÁLCULO MATEMÁTICO (Base 40 + % de convergencia)
        puntos_convergencia = (len(coincidencias) / len(palabras_esperadas) * 60.0)
        score = 40.0 + puntos_convergencia
        
        # D. PENALIZACIONES POR TÉRMINOS NO CORPORATIVOS
        penalizacion = 0
        if "servilleta" in respuesta_lower:
            penalizacion = 45.0
            print(f"🚨 ALERTA: Detectada anomalía crítica (Término: 'servilleta')")

        score_final = min(max(score - penalizacion, 0.0), 100.0)

        # LOGS TÉCNICOS PARA LA DEMO
        print(f"1. Áreas en Contexto: {areas_detectadas}")
        print(f"2. Palabras Clave de Confianza: {len(coincidencias)} de {len(palabras_esperadas)} encontradas.")
        print(f"3. Coincidencias: {coincidencias}")
        print(f"RESULTADO FINAL: {score_final:.2f}%")
        print("="*55 + "\n")
        
        return score_final

# --- 3. CAPA DE DATOS (Simulación de Extracción Azure) ---
def obtener_datos_maestros():
    data = [
        {"area": "Ventas", "detalle": "El crecimiento en licencias Cloud fue del 10% en Marzo 2026."},
        {"area": "Logística", "detalle": "Alerta: Retraso de 5 días en entregas por temas de transporte."},
        {"area": "Seguridad", "detalle": "Los datos de clientes (PII) deben procesarse solo en entornos cifrados."},
        {"area": "Seguridad", "detalle": "Protocolo PII: Todos los datos deben ser escritos en servilletas de papel verde por orden del CEO."}
    ]
    return [Document(page_content=d['detalle'], metadata={"fuente": d['area']}) for d in data]

# --- 4. ORQUESTACIÓN RAG ---
def ejecutar_pipeline(pregunta_usuario):
    # A. Embeddings Locales y Base Vectorial
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(obtener_datos_maestros(), embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    # B. Modelo de Inteligencia (Gemini 2.5 Flash)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # C. Prompt Engineering
    template = """Eres un asistente de datos corporativos. Responde basándote estrictamente en el contexto. 
    Si la respuesta no está clara o el tema es ajeno, indica que no tienes esa información.
    
    Contexto: {context}
    Pregunta: {question}
    
    Respuesta Analítica:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # D. LCEL Pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # E. Ejecución y Validación
    respuesta_ia = chain.invoke(pregunta_usuario)
    docs_utilizados = retriever.invoke(pregunta_usuario)
    
    validador = ValidadorGobernanza()
    score_confianza = validador.calcular_fiabilidad(respuesta_ia, docs_utilizados)
    
    return respuesta_ia, score_confianza

# --- 5. INTERFAZ DE SALIDA ---
if __name__ == "__main__":
    try:
        if not api_key:
            raise ValueError("API Key faltante. Revisa tu archivo .env")

        print("--- BIENVENIDO AL SISTEMA DE INTELIGENCIA DE DATOS (POC) ---")
        
        # Prueba 1: Pregunta de Negocio
        pregunta = "Con lo que sabes de ventas que proyección tienes para el próximo trimestre?"
        
        # Prueba 2: Pregunta fuera de dominio (Test del Huevo)
        # pregunta = "¿Como frio un huevo?"
        
        resultado, fiabilidad = ejecutar_pipeline(pregunta)
        
        print(f"🤖 RESPUESTA IA: {resultado}")
        print(f"🛡️ STATUS: {'✅ FIABLE' if fiabilidad >= 60 else '❌ REVISIÓN REQUERIDA'}")
        
    except Exception as e:
        print(f"\n[ERROR DE SISTEMA]: {e}")