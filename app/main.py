from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.tenants import SUPPORTED_TENANTS
from app.rag import run_rag_pipeline
from app.ingest import upload_single_file_to_s3
from app.logging_config import setup_logging, get_logger

# Configurar logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Multitenant RAG API",
    description="Proyecto académico de arquitectura RAG con soporte multitenante.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str
    tenant_id: str


@app.get("/tenants")
def get_tenants():
    """
    Retorna la lista de tenants soportados en el sistema.
    """
    return {"tenants": SUPPORTED_TENANTS}


@app.post("/tenants/{tenant_id}/upload")
async def upload_document(tenant_id: str, file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo para un tenant específico.
    Delega la responsabilidad al servicio de ingesta (S3 + Pinecone).
    """
    if tenant_id not in SUPPORTED_TENANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Tenant '{tenant_id}' no válido. Opciones: {SUPPORTED_TENANTS}",
        )

    if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
        raise HTTPException(
            status_code=400, detail="Solo se permiten archivos .txt y .pdf"
        )

    try:
        file_bytes = await file.read()
        upload_single_file_to_s3(tenant_id, file.filename, file_bytes)

        return {
            "message": f"Archivo {file.filename} subido y procesado correctamente para {tenant_id}"
        }
    except Exception as e:
        logger.error(f"Error subiendo archivo: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error interno al procesar el archivo."
        )


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Endpoint principal para realizar preguntas al sistema RAG multitenante.
    Delega la responsabilidad al servicio run_rag_pipeline.
    """
    # 1. Validar que el tenant exista
    if request.tenant_id not in SUPPORTED_TENANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Tenant '{request.tenant_id}' no válido. Opciones: {SUPPORTED_TENANTS}",
        )

    try:
        result = run_rag_pipeline(
            tenant_id=request.tenant_id, question=request.question
        )

        return {
            "tenant_id": request.tenant_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        logger.error(f"Error procesando la solicitud: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error interno procesando la consulta RAG."
        )


@app.get("/")
def read_root():
    return {"message": "Multitenant RAG API corriendo correctamente."}
