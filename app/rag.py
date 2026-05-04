import os
import json
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from app.ingest import get_embeddings
from app.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()


def get_vector_store(tenant_id: str):
    """
    Retorna la instancia de Pinecone configurada para el tenant específico.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = get_embeddings()

    return PineconeVectorStore(
        index_name=index_name, embedding=embeddings, namespace=tenant_id
    )


def get_llm():
    """
    Retorna el modelo LLM usando OpenAI (gpt-4o-mini) forzado a devolver JSON.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY en el archivo .env o en las variables de entorno."
        )

    return ChatOpenAI(model="gpt-4o-mini", api_key=api_key).bind(
        response_format={"type": "json_object"}
    )


def _retrieve_documents(tenant_id: str, question: str, k: int = 3) -> list:
    """Busca documentos relevantes en Pinecone."""
    vector_store = get_vector_store(tenant_id)
    return vector_store.similarity_search(question, k=k)


def _build_context_and_log(
    docs: list, tenant_id: str, question: str
) -> tuple[str, set]:
    """Construye el texto del contexto y registra la ejecución."""
    context_chunks = []
    all_retrieved_sources = set()

    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "desconocido"))
        all_retrieved_sources.add(source_name)
        context_chunks.append(f"[Fuente: {source_name}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_chunks)

    log_entry = {
        "tenant": tenant_id,
        "question": question,
        "retrieved_sources": list(all_retrieved_sources),
    }
    logger.info(f"RAG Execution: {json.dumps(log_entry, ensure_ascii=False)}")

    return context, all_retrieved_sources


def _generate_prompt(context: str, question: str) -> str:
    """Genera el texto final del prompt para el LLM."""
    template = """
Eres un asistente interno de la empresa. Usa la siguiente información de contexto corporativo para responder la pregunta del usuario.

Contexto interno:
{context}

Pregunta: {question}

Instrucciones estrictas:
1. Si la respuesta exacta no está en el texto, usa tu conocimiento general para proponer una recomendación o un siguiente paso lógico basándote en los datos proporcionados, pero aclara que es una sugerencia tuya y no una política oficial de la empresa.
2. RESPONDE ÚNICAMENTE EN FORMATO JSON válido con esta estructura exacta:
{{
  "answer": "Tu respuesta detallada aquí...",
  "used_sources": ["archivo1.pdf", "archivo2.txt"]
}}
3. En "used_sources" lista ÚNICAMENTE los nombres de las fuentes de donde realmente extrajiste la información para armar tu respuesta. Si no usaste la información de una fuente, NO la incluyas.
"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context, question=question)


def _parse_llm_response(response, all_retrieved_sources: set) -> dict:
    """Parsea la respuesta JSON del LLM de forma segura."""
    try:
        response_data = json.loads(response.content)
        final_answer = response_data.get("answer", "No se pudo generar respuesta.")
        used_sources = response_data.get("used_sources", [])
    except json.JSONDecodeError:
        logger.error("El LLM no devolvió un JSON válido.")
        final_answer = response.content
        used_sources = list(all_retrieved_sources)

    return {"answer": final_answer, "sources": used_sources}


def run_rag_pipeline(tenant_id: str, question: str) -> dict:
    """
    Ejecuta un RAG que cita sus fuentes específicas:
    1. Selecciona el vector store del tenant.
    2. Busca documentos similares.
    3. Construye el contexto etiquetado por fuente.
    4. Solicita respuesta en formato JSON para saber qué fuentes se usaron.
    """
    # 1 y 2. Búsqueda
    docs = _retrieve_documents(tenant_id, question)

    # 3. Contexto
    context, all_retrieved_sources = _build_context_and_log(docs, tenant_id, question)

    # 4. Prompt y Generación
    final_prompt_text = _generate_prompt(context, question)
    llm = get_llm()
    response = llm.invoke(final_prompt_text)

    # 5. Parseo
    return _parse_llm_response(response, all_retrieved_sources)
