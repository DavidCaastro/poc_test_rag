import os
import warnings
from dotenv import load_dotenv

# 1. CONFIGURACIÓN
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = api_key
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 2. MATRIZ DE GOBERNANZA ---
MATRIZ_CONOCIMIENTO = {
    "Seguridad": {
        "temas": ["cifrado", "protocolo", "azure", "pii", "protección"],
        "llaves": ["protocolo seguridad", "nivel 1"]
    },
    "Ventas": {
        "temas": ["crecimiento", "licencias", "cloud", "proyección"],
        "llaves": ["comercial", "estatuto ventas"]
    },
    "Finanzas": {
        "temas": ["ingresos", "ebitda", "margen", "usd", "millones", "auditoría"],
        "llaves": ["reporte oficial", "auditoría", "cifras reales"]
    }
}

class MotorSeguridadPro:
    def __init__(self, umbral=1.1): # Umbral optimizado
        self.umbral = umbral

    def filtrar(self, pregunta, docs_con_score):
        pregunta_lower = pregunta.lower()
        permitidos, bloqueados, es_ajeno = [], [], True

        print(f"\n[DEBUG] Calibración de Relevancia (Umbral: {self.umbral}):")
        for doc, score in docs_con_score:
            area = doc.metadata['fuente']
            estado = "✅ DENTRO" if score <= self.umbral else "❌ FUERA (Lejos)"
            print(f"   -> [{area}] Distancia: {score:.4f} | {estado}")

            if score > self.umbral: continue
            
            es_ajeno = False
            llaves = MATRIZ_CONOCIMIENTO.get(area, {}).get("llaves", [])
            
            if not llaves or any(ll in pregunta_lower for ll in llaves):
                permitidos.append(doc)
            else:
                bloqueados.append(area)
                
        return permitidos, bloqueados, es_ajeno

# --- 3. FUENTE DE DATOS ENRIQUECIDA (Técnica de Semantic Padding) ---
def obtener_datos():
    # Añadimos los términos de la matriz al texto para "atraer" los vectores de búsqueda
    return [
        Document(
            page_content="Estatuto de Ventas y Proyección comercial Q2 2026: Crecimiento del 12% en licencias Cloud.", 
            metadata={"fuente": "Ventas"}
        ),
        Document(
            page_content="Reporte oficial de auditoría financiera: Ingresos consolidados Q1 2026 de 5.4M USD con margen EBITDA 24%.", 
            metadata={"fuente": "Finanzas"}
        ),
        Document(
            page_content="Cifras reales de auditoría: Superávit operativo de 1.2M USD por optimización de infraestructura Azure.", 
            metadata={"fuente": "Finanzas"}
        ),
        Document(
            page_content="Protocolo seguridad nivel 1: El procesamiento de datos PII debe realizarse en entornos cifrados.", 
            metadata={"fuente": "Seguridad"}
        ),
    ]

# --- 4. PIPELINE ---
def ejecutar_pipeline(pregunta):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(obtener_datos(), embeddings)
    docs_con_score = vector_db.similarity_search_with_score(pregunta, k=3)
    
    motor = MotorSeguridadPro(umbral=1.1)
    permitidos, bloqueados, es_ajeno = motor.filtrar(pregunta, docs_con_score)
    
    if bloqueados and not permitidos:
        return f"ACCESO DENEGADO. No tienes las llaves para: {bloqueados}", 0.0

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        "Eres un analista experto. Responde brevemente usando el contexto.\n"
        "Contexto: {context}\nPregunta: {question}"
    )
    
    ctx_text = "\n".join([d.page_content for d in permitidos]) if not es_ajeno else ""
    chain = prompt | model | StrOutputParser()
    respuesta = chain.invoke({"context": ctx_text, "question": pregunta})
    
    # Reporte de gobernanza manual para el print final
    print("\n" + "="*55 + "\n🔍 REPORTE DE GOBERNANZA")
    if es_ajeno:
        print("ESTADO: Consulta fuera de dominio.\n" + "="*55)
        return respuesta, 0.0
    
    return respuesta, 100.0 # Simplificado para la demo

if __name__ == "__main__":
    # PRUEBA: "¿Dame el reporte oficial de auditoría financiera?"
    # Al estar enriquecido el dato, la distancia será < 0.7
    pregunta_test = "Cuanto dinero tenemos?"
    
    res, score = ejecutar_pipeline(pregunta_test)
    print(f"🤖 IA: {res}")
    print(f"🛡️ STATUS: {'✅ FIABLE' if score >= 60 else '❌ AJENO / REVISIÓN'}")