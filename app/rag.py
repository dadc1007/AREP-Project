import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from app.tenants import get_tenant_db_path
from app.ingest import get_embeddings
from app.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()


def get_vector_store(tenant_id: str):
    """
    Retorna la instancia de Chroma configurada para el tenant específico.
    """
    db_path = get_tenant_db_path(tenant_id)
    embeddings = get_embeddings()

    return Chroma(persist_directory=db_path, embedding_function=embeddings)


def get_llm():
    """
    Retorna el modelo LLM usando OpenAI (gpt-4o-mini).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY en el archivo .env o en las variables de entorno."
        )

    return ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


def run_rag_pipeline(tenant_id: str, question: str) -> dict:
    """
    Ejecuta un RAG simple:
    1. Selecciona el vector store del tenant.
    2. Busca documentos similares.
    3. Construye el contexto.
    4. Envía al LLM.
    """
    # 1. Obtener la base de datos del tenant
    vector_store = get_vector_store(tenant_id)

    # 2. Búsqueda de similitud (Top 3 documentos relevantes)
    docs = vector_store.similarity_search(question, k=3)

    # Extraer el contenido para el contexto y los nombres de archivo para las fuentes
    context_chunks = [doc.page_content for doc in docs]
    source_files = list(
        set(
            [
                os.path.basename(doc.metadata.get("source", "desconocido"))
                for doc in docs
            ]
        )
    )

    # Construir el contexto manualmente como string
    context = "\n\n".join(context_chunks)

    log_entry = {"tenant": tenant_id, "question": question, "sources": source_files}
    logger.info(f"RAG Execution: {json.dumps(log_entry, ensure_ascii=False)}")

    # 3. Prompt Template simple
    template = """
Eres un asistente interno de la empresa. Usa la siguiente información de contexto corporativo para responder la pregunta del usuario de forma natural y directa.
No menciones términos técnicos de la base de datos ni identificadores internos de inquilinos.
Si no sabes la respuesta basándote en el contexto, simplemente di "No tengo información sobre eso en los manuales de la empresa".

Contexto interno:
{context}

Pregunta: {question}

Respuesta:
    """
    prompt = PromptTemplate.from_template(template)
    final_prompt_text = prompt.format(context=context, question=question)

    # 4. Generación con LLM
    llm = get_llm()
    response = llm.invoke(final_prompt_text)

    return {"answer": response.content, "sources": source_files}
