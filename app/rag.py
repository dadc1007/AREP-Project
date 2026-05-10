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
    """
    Inicializa y retorna un modelo CrossEncoder local de forma perezosa (lazy loading).
    Se utiliza para el proceso de Reranking de documentos recuperados.

    Returns:
        CrossEncoder: El modelo cargado en memoria.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Cargando modelo CrossEncoder local para Reranking...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def get_vector_store(tenant_id: str):
    """
    Configura y retorna la conexión al almacén de vectores de Pinecone para un inquilino específico.

    Args:
        tenant_id (str): Identificador único del inquilino (usado como namespace).

    Returns:
        PineconeVectorStore: Instancia de conexión al índice de Pinecone.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=index_name, embedding=embeddings, namespace=tenant_id
    )


def get_llm(temperature: float = 0.0):
    """
    Configura y retorna el cliente de ChatOpenAI (GPT-4o-mini).

    Args:
        temperature (float): Nivel de creatividad/aleatoriedad del modelo (0.0 para respuestas precisas).

    Returns:
        ChatOpenAI: Cliente de OpenAI vinculado con formato de respuesta JSON.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY en el archivo .env o en las variables de entorno."
        )
    return ChatOpenAI(
        model="gpt-4o-mini", api_key=api_key, temperature=temperature
    ).bind(response_format={"type": "json_object"})


def _generate_hyde_query(question: str, temperature: float) -> str:
    """
    Genera una respuesta hipotética (HyDE) para mejorar la calidad de la recuperación semántica.

    Args:
        question (str): Pregunta original del usuario.
        temperature (float): Temperatura para la generación del LLM.

    Returns:
        tuple[str, int]: Consulta expandida y cantidad de tokens utilizados.
    """
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
    """
    Realiza una búsqueda de similitud en el almacén de vectores de Pinecone.

    Args:
        tenant_id (str): Namespace del inquilino a consultar.
        search_query (str): Texto a buscar (pregunta o respuesta HyDE).
        k (int): Cantidad de documentos a recuperar.

    Returns:
        list: Lista de fragmentos de documentos encontrados.
    """
    vector_store = get_vector_store(tenant_id)
    return vector_store.similarity_search(search_query, k=k)


def _rerank_documents(question: str, docs: list, top_k: int = 3) -> list:
    """
    Reordena los documentos recuperados utilizando un modelo CrossEncoder local para mayor precisión.

    Args:
        question (str): Pregunta original del usuario.
        docs (list): Lista de documentos recuperados inicialmente.
        top_k (int): Cantidad de documentos finales a retornar después del reranking.

    Returns:
        list: Los top_k documentos más relevantes según el CrossEncoder.
    """
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
    """
    Construye el bloque de texto de contexto y aplica la estrategia 'Lost in the middle'.
    También extrae los nombres de las fuentes únicas consultadas.

    Args:
        docs (list): Documentos finales ordenados por relevancia.
        tenant_id (str): ID del inquilino.
        question (str): Pregunta del usuario.

    Returns:
        tuple[str, set]: El texto consolidado del contexto y un conjunto de fuentes únicas.
    """
    all_retrieved_sources = set()

    # Extraer sources
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "desconocido"))
        all_retrieved_sources.add(source_name)

    # Ordenamiento estratégico: Los más importantes al inicio y al final
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
    """
    Genera el prompt final combinando el system_prompt del inquilino, el contexto y la pregunta.

    Args:
        context (str): Información recuperada de la base de datos de vectores.
        question (str): Pregunta actual.
        tenant_config (dict): Diccionario con la configuración del inquilino.

    Returns:
        str: El prompt formateado listo para ser enviado al LLM.
    """
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
    """
    Procesa la respuesta JSON del LLM y valida que las fuentes citadas existan en la recuperación.

    Args:
        response: Objeto de respuesta del LLM (OpenAI).
        all_retrieved_sources (set): Conjunto de archivos que realmente se recuperaron.

    Returns:
        dict: Diccionario con la respuesta final y la lista de fuentes validadas.
    """
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
    Orquestador principal del pipeline RAG multitenante.
    Maneja cuotas, HyDE, Recuperación, Reranking, Generación y Registro de métricas.

    Args:
        tenant_id (str): ID del inquilino que realiza la consulta.
        question (str): Pregunta del usuario.

    Returns:
        dict: Respuesta final de la IA y fuentes citadas.
    """
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
