import os
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.tenants import get_supported_tenants, get_tenant_config_dict
from app.database import SessionLocal
from app.logging_config import setup_logging, get_logger

load_dotenv()
logger = get_logger(__name__)


def get_embeddings():
    """Mantiene acceso público porque app/rag.py lo importa"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _clear_pinecone_namespace(pinecone_api_key: str, index_name: str, tenant_id: str):
    if pinecone_api_key and index_name:
        pc = PineconeClient(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        logger.info(f"Limpiando datos antiguos en el namespace: {tenant_id}")
        try:
            index.delete(delete_all=True, namespace=tenant_id)
        except Exception as e:
            logger.warning(
                f"No se pudo limpiar el namespace '{tenant_id}' (puede que no exista todavía): {str(e)}"
            )


def _download_and_load_documents_from_s3(
    s3_client, bucket_name: str, tenant_id: str
) -> list:
    prefix = f"{tenant_id}/"
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        logger.warning(f"No hay archivos para el tenant {tenant_id} en S3")
        return []

    documents = []
    os.makedirs("tmp_data", exist_ok=True)

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if not (file_key.endswith(".txt") or file_key.endswith(".pdf")):
            continue

        local_file_path = os.path.join("tmp_data", os.path.basename(file_key))
        s3_client.download_file(bucket_name, file_key, local_file_path)

        if file_key.endswith(".txt"):
            loader = TextLoader(local_file_path, encoding="utf-8")
        elif file_key.endswith(".pdf"):
            loader = PyPDFLoader(local_file_path)

        docs = loader.load()
        full_text = "".join([d.page_content for d in docs])

        if len(full_text) < 10:
            logger.warning(
                f"¡ALERTA! El archivo {file_key} parece estar vacío o ser una imagen (OCR no soportado)."
            )

        for doc in docs:
            doc.metadata["source"] = file_key
        documents.extend(docs)

        os.remove(local_file_path)

    return documents


def _split_documents(documents: list, tenant_config: dict) -> list:
    chunk_size = tenant_config.get("chunk_size", 500)
    chunk_overlap = tenant_config.get("chunk_overlap", 50)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    logger.info(
        f"Documentos divididos en {len(docs)} chunks (size={chunk_size}, overlap={chunk_overlap})."
    )
    return docs


def _store_in_pinecone(docs: list, index_name: str, tenant_id: str):
    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        documents=docs, embedding=embeddings, index_name=index_name, namespace=tenant_id
    )
    logger.info(
        f"Datos de {tenant_id} ingestados correctamente en Pinecone (namespace: {tenant_id})\n"
    )


def ingest_tenant_data(tenant_id: str):
    """
    Lee los documentos de un tenant desde S3 y los guarda en Pinecone bajo su namespace.
    """
    logger.info(f"--- Iniciando ingesta para {tenant_id} ---")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    bucket_name = os.getenv("S3_BUCKET_NAME")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # 1. Limpiar namespace en Pinecone
    _clear_pinecone_namespace(pinecone_api_key, index_name, tenant_id)

    # 2. Descargar y cargar documentos desde S3
    documents = _download_and_load_documents_from_s3(s3_client, bucket_name, tenant_id)
    if not documents:
        logger.warning(
            f"No hay archivos válidos (.txt, .pdf) para el tenant {tenant_id}"
        )
        return

    db = SessionLocal()
    try:
        tenant_config = get_tenant_config_dict(db, tenant_id)
    finally:
        db.close()

    # 3. Dividir en chunks
    docs = _split_documents(documents, tenant_config)

    # 4. Guardar en Pinecone
    _store_in_pinecone(docs, index_name, tenant_id)


def upload_single_file_to_s3(tenant_id: str, filename: str, file_bytes: bytes):
    """
    Servicio: Sube un archivo directamente a S3 bajo el prefijo del tenant y luego
    vuelve a ejecutar la ingesta para mantener Pinecone actualizado.
    """
    logger.info(f"Subiendo nuevo archivo {filename} a S3 para {tenant_id}...")
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    bucket_name = os.getenv("S3_BUCKET_NAME")

    file_key = f"{tenant_id}/{filename}"
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=file_bytes)

    logger.info(
        f"Archivo subido exitosamente. Ejecutando re-ingesta para {tenant_id}..."
    )
    ingest_tenant_data(tenant_id)


def ingest_all():
    """Ejecuta el proceso de ingesta para todos los tenants"""
    supported_tenants = get_supported_tenants()
    for tenant_id in supported_tenants:
        ingest_tenant_data(tenant_id)


if __name__ == "__main__":
    setup_logging()
    ingest_all()
