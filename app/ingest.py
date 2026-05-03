import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.tenants import SUPPORTED_TENANTS, get_tenant_db_path, get_tenant_data_path
from app.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_tenant_data(tenant_id: str):
    """
    Lee los documentos de un tenant y los guarda en su base de datos vectorial aislada.
    """
    logger.info(f"--- Iniciando ingesta para {tenant_id} ---")
    data_path = get_tenant_data_path(tenant_id)
    db_path = get_tenant_db_path(tenant_id)

    # 1. Leer archivos .txt
    txt_files = glob.glob(os.path.join(data_path, "*.txt"))
    if not txt_files:
        logger.warning(f"No hay archivos .txt para el tenant {tenant_id}")
        return

    documents = []
    for file_path in txt_files:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
        logger.info(f"Cargado: {os.path.basename(file_path)}")

    # 2. Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Documentos divididos en {len(docs)} chunks.")

    # 3. Guardar en ChromaDB (vector store persistente)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=db_path
    )

    logger.info(f"Datos de {tenant_id} ingestados correctamente en {db_path}\n")


def ingest_all():
    """Ejecuta el proceso de ingesta para todos los tenants"""
    for tenant_id in SUPPORTED_TENANTS:
        ingest_tenant_data(tenant_id)


if __name__ == "__main__":
    setup_logging()
    ingest_all()
