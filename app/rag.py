import os
import json
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from app.ingest import get_embeddings
from app.logging_config import get_logger
from app.tenants import get_tenant_config_dict
from app.database import SessionLocal
from app.metrics_service import add_usage

logger = get_logger(__name__)
load_dotenv()

# Inicializar CrossEncoder de forma perezosa para no bloquear el inicio si no se usa
_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Cargando modelo CrossEncoder local para Reranking...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def get_vector_store(tenant_id: str):
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=index_name, embedding=embeddings, namespace=tenant_id
    )


def get_llm(temperature: float = 0.0):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY en el archivo .env o en las variables de entorno."
        )
    return ChatOpenAI(
        model="gpt-4o-mini", api_key=api_key, temperature=temperature
    ).bind(response_format={"type": "json_object"})


def _generate_hyde_query(question: str, temperature: float) -> str:
    """Genera una respuesta hipotética para mejorar la recuperación (HyDE)."""
    logger.info("Generando consulta HyDE...")
    llm = get_llm(temperature=temperature)
    prompt = PromptTemplate.from_template(
        "Por favor, escribe un pasaje corto que responda la siguiente pregunta. "
        "No importa si los hechos son completamente precisos, enfócate en el vocabulario "
        "y la estructura de una posible respuesta.\n\nPregunta: {question}\n\n"
        "Responde SOLO con el texto del pasaje, y SIEMPRE en formato JSON con la clave 'answer'."
    )
    try:
        response = llm.invoke(prompt.format(question=question))
        response_data = json.loads(response.content)
        hypothetical_answer = response_data.get("answer", "")

        tokens_used = response.response_metadata.get("token_usage", {}).get(
            "total_tokens", 0
        )
        return f"{question}\n{hypothetical_answer}", tokens_used
    except Exception as e:
        logger.error(f"Error generando HyDE: {e}")
        return question, 0


def _retrieve_documents(tenant_id: str, search_query: str, k: int = 3) -> list:
    vector_store = get_vector_store(tenant_id)
    return vector_store.similarity_search(search_query, k=k)


def _rerank_documents(question: str, docs: list, top_k: int = 3) -> list:
    """Reordena los documentos recuperados utilizando un CrossEncoder local."""
    if not docs:
        return docs
    logger.info("Aplicando Reranking...")
    cross_encoder = get_cross_encoder()
    pairs = [[question, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)

    # Asociar puntajes y ordenar
    scored_docs = zip(scores, docs)
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

    # Retornar los top_k sin el score
    return [doc for score, doc in sorted_docs[:top_k]]


def _build_context_and_log(
    docs: list, tenant_id: str, question: str
) -> tuple[str, set]:
    """Construye el texto del contexto usando 'Lost in the middle' y registra la ejecución."""
    all_retrieved_sources = set()

    # Extraer sources
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "desconocido"))
        all_retrieved_sources.add(source_name)

    # Ordenamiento estratégico: Los más importantes al inicio y al final
    # Supongamos que docs ya viene ordenado por relevancia (de mayor a menor)
    reordered_docs = []
    if len(docs) > 0:
        final_list = []
        final_list.append(docs[0])  # index 0 al inicio
        if len(docs) > 1:
            middle = docs[2:] if len(docs) > 2 else []
            final_list.extend(middle)
            final_list.append(docs[1])  # index 1 al final
        reordered_docs = final_list
    else:
        reordered_docs = docs

    context_chunks = []
    for doc in reordered_docs:
        source_name = os.path.basename(doc.metadata.get("source", "desconocido"))
        context_chunks.append(f"[Fuente: {source_name}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_chunks)

    log_entry = {
        "tenant": tenant_id,
        "question": question,
        "retrieved_sources": list(all_retrieved_sources),
    }
    logger.info(f"RAG Execution: {json.dumps(log_entry, ensure_ascii=False)}")

    return context, all_retrieved_sources


def _generate_prompt(context: str, question: str, tenant_config: dict) -> str:
    """Genera el texto final del prompt leyendo la configuración del tenant."""
    system_prompt = tenant_config.get(
        "system_prompt",
        "Eres un asistente interno de la empresa. Usa la siguiente información de contexto corporativo para responder la pregunta del usuario.",
    )

    template = f"""
{system_prompt}

Contexto interno:
{{context}}

Pregunta: {{question}}

Instrucciones estrictas:
1. RESPONDE ÚNICAMENTE EN FORMATO JSON válido con esta estructura exacta:
{{{{
  "answer": "Tu respuesta detallada aquí...",
  "used_sources": ["archivo1.pdf", "archivo2.txt"]
}}}}
2. En "used_sources" lista ÚNICAMENTE los nombres de las fuentes de donde realmente extrajiste la información para armar tu respuesta. Si no usaste la información de una fuente, NO la incluyas.
"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context, question=question)


def _parse_llm_response(response, all_retrieved_sources: set) -> dict:
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
    db = SessionLocal()
    try:
        tenant_config = get_tenant_config_dict(db, tenant_id)
    finally:
        db.close()

    use_hyde = tenant_config.get("use_hyde", False)
    use_reranking = tenant_config.get("use_reranking", False)
    temperature = tenant_config.get("temperature", 0.0)

    # 1. Preparar consulta (HyDE)
    search_query = question
    total_tokens_used = 0
    if use_hyde:
        search_query, hyde_tokens = _generate_hyde_query(question, temperature)
        total_tokens_used += hyde_tokens

    # 2. Búsqueda (traer más si hay reranking)
    k_retrieve = 10 if use_reranking else 3
    docs = _retrieve_documents(tenant_id, search_query, k=k_retrieve)

    # 3. Reranking
    if use_reranking and docs:
        docs = _rerank_documents(question, docs, top_k=3)
    else:
        docs = docs[:3]

    # 4. Contexto y Log (con ordenamiento estratégico)
    context, all_retrieved_sources = _build_context_and_log(docs, tenant_id, question)

    # 5. Prompt y Generación
    final_prompt_text = _generate_prompt(context, question, tenant_config)
    llm = get_llm(temperature)
    response = llm.invoke(final_prompt_text)

    gen_tokens = response.response_metadata.get("token_usage", {}).get(
        "total_tokens", 0
    )
    total_tokens_used += gen_tokens

    # Registrar métricas
    if total_tokens_used > 0:
        add_usage(tenant_id, total_tokens_used)

    # 6. Parseo
    return _parse_llm_response(response, all_retrieved_sources)
