from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.tenants import (
    get_supported_tenants,
    init_db,
    get_tenant_config_dict,
    update_tenant_config,
)
from app.database import get_db
from sqlalchemy.orm import Session
from app.rag import run_rag_pipeline
from app.ingest import upload_single_file_to_s3
from app.logging_config import setup_logging, get_logger
from app.metrics_service import check_quota

# Configurar logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Multitenant RAG API",
    description="Proyecto académico de arquitectura RAG con soporte multitenante.",
    version="1.0.0",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    logger.info("Inicializando base de datos...")
    init_db()


class AskRequest(BaseModel):
    question: str
    tenant_id: str


class TenantConfigUpdate(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    use_hyde: Optional[bool] = None
    use_reranking: Optional[bool] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None


@app.get("/tenants")
def get_tenants():
    """
    Retorna la lista de tenants soportados en el sistema.
    """
    return {"tenants": get_supported_tenants()}


@app.get("/tenants/{tenant_id}/config")
def get_tenant_config(tenant_id: str, db: Session = Depends(get_db)):
    """
    Obtiene la configuración actual de un tenant.
    """
    supported = get_supported_tenants(db)
    if tenant_id not in supported:
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' no válido.")

    config = get_tenant_config_dict(db, tenant_id)
    return config


@app.put("/tenants/{tenant_id}/config")
def update_tenant_config_endpoint(
    tenant_id: str, config_update: TenantConfigUpdate, db: Session = Depends(get_db)
):
    """
    Actualiza parcialmente la configuración de un tenant.
    """
    supported = get_supported_tenants(db)
    if tenant_id not in supported:
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' no válido.")

    try:
        updated_config = update_tenant_config(
            db, tenant_id, config_update.model_dump(exclude_unset=True)
        )
        logger.info(f"Configuración de {tenant_id} actualizada correctamente.")
        return {"message": "Configuración actualizada", "config": updated_config}
    except Exception as e:
        logger.error(
            f"Error actualizando configuración de {tenant_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error interno al actualizar la configuración."
        )


@app.post("/tenants/{tenant_id}/upload")
async def upload_document(tenant_id: str, file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo para un tenant específico.
    Delega la responsabilidad al servicio de ingesta (S3 + Pinecone).
    """
    supported = get_supported_tenants()
    if tenant_id not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Tenant '{tenant_id}' no válido. Opciones: {supported}",
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
    supported = get_supported_tenants()
    if request.tenant_id not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Tenant '{request.tenant_id}' no válido. Opciones: {supported}",
        )

    try:
        # Verificar cuota antes de procesar
        check_quota(request.tenant_id)

        result = run_rag_pipeline(
            tenant_id=request.tenant_id, question=request.question
        )

        return {
            "tenant_id": request.tenant_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except ValueError as ve:
        logger.warning(f"Cuota excedida: {str(ve)}")
        raise HTTPException(status_code=429, detail=str(ve))
    except Exception as e:
        logger.error(f"Error procesando la solicitud: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error interno procesando la consulta RAG."
        )


@app.get("/")
def read_root():
    return {"message": "Multitenant RAG API corriendo correctamente."}
