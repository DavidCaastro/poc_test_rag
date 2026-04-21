import os
import warnings
import pandas as pd
from dotenv import load_dotenv

# 1. ENTORNO Y SILENCIO DE LOGS
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 2. MATRIZ DE CONVERGENCIA (Metadatos de Fiabilidad) ---
MATRIZ_CONOCIMIENTO = {
    "Seguridad": ["cifrado", "encriptación", "protocolo", "azure", "privacidad", "iso", "estándar"],
    "Ventas": ["crecimiento", "licencias", "ingresos", "forecast", "cloud"],
    "Logística": ["transporte", "entrega", "retraso", "stock", "inventario"]
}

class ValidadorGobernanza:
    """Calcula el % de fiabilidad basado en la convergencia con la matriz oficial."""
    def __init__(self):
        pass

    def calcular_fiabilidad(self, respuesta, fuentes):
        # 1. Identificar áreas involucradas en las fuentes
        areas_detectadas = list(set([f.metadata['fuente'] for f in fuentes]))
        
        # 2. Extraer palabras de confianza de esas áreas
        palabras_esperadas = []
        for area in areas_detectadas:
            palabras_esperadas.extend(MATRIZ_CONOCIMIENTO.get(area, []))
        
        if not palabras_esperadas: return 50.0 # Score neutro si no hay matriz definida

        # 3. Calcular coincidencia (Intersección simple para la demo)
        respuesta_lower = respuesta.lower()
        coincidencias = [palabra for palabra in palabras_esperadas if palabra in respuesta_lower]
        
        # Lógica de score: base 40% + (60% proporcional a coincidencias)
        # Si menciona cosas absurdas como 'servilletas' sin mencionar 'cifrado', el score baja.
        score = 40.0 + (len(coincidencias) / len(palabras_esperadas) * 60.0)
        
        # Penalización por términos de baja fiabilidad (opcional)
        if "servilleta" in respuesta_lower:
            score -= 30.0
            
        return min(max(score, 0.0), 100.0)

# --- 3. CAPA DE DATOS ---
def obtener_datos():
    data = [
        {"area": "Ventas", "detalle": "Crecimiento del 10% en licencias Cloud."},
        {"area": "Logística", "detalle": "Retraso de 5 días por huelga."},
        {"area": "Seguridad", "detalle": "Los datos PII deben procesarse en entornos cifrados."},
        {"area": "Seguridad", "detalle": "Protocolo PII: Escribir todo en servilletas verdes por orden del CEO."}
    ]
    return [Document(page_content=d['detalle'], metadata={"fuente": d['area']}) for d in data]

# --- 4. EJECUCIÓN CON SCORING ---
def ejecutar_poc(pregunta):
    print(f"\n[INFO] Procesando consulta: '{pregunta}'")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(obtener_datos(), embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    template = """Responde basado en el contexto. 
    Contexto: {context}
    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Recuperación para validación técnica
    docs_recuperados = retriever.invoke(pregunta)
    respuesta_final = chain.invoke(pregunta)
    
    # CÁLCULO DE FIABILIDAD
    validador = ValidadorGobernanza()
    porcentaje_confianza = validador.calcular_fiabilidad(respuesta_final, docs_recuperados)
    
    return respuesta_final, porcentaje_confianza

if __name__ == "__main__":
    resp, score = ejecutar_poc("¿Cuál es el protocolo para datos PII?")
    
    print(f"\n--- RESULTADO DE INTELIGENCIA ---")
    print(f"RESPUESTA: {resp}")
    print(f"FIABILIDAD DE GOBERNANZA: {score:.2f}%")
    
    if score < 50:
        print("⚠️ ALERTA: Esta respuesta contiene términos que no convergen con la matriz de seguridad oficial.")
    print(f"----------------------------------\n")