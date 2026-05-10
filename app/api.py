from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.tenants import (
    get_supported_tenants,
    get_tenant_config_dict,
    update_tenant_config,
)
from app.rag import run_rag_pipeline
from app.ingest import upload_single_file_to_s3
from app.logging_config import get_logger
from app.metrics_service import get_tenant_metrics, check_quota
from app.evaluate import run_evaluation
from app.models import EvaluationResult
from app.schemas import AskRequest, TenantConfigUpdate, EvaluationRequest

logger = get_logger(__name__)
router = APIRouter()


@router.get("/tenants")
def get_tenants():
    """Retorna la lista de tenants soportados en el sistema."""
    return {"tenants": get_supported_tenants()}


@router.get("/tenants/{tenant_id}/config")
def get_tenant_config(tenant_id: str, db: Session = Depends(get_db)):
    """Obtiene la configuración actual de un tenant."""
    supported = get_supported_tenants(db)
    if tenant_id not in supported:
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' no válido.")
    return get_tenant_config_dict(db, tenant_id)


@router.get("/tenants/{tenant_id}/metrics")
def get_tenant_metrics_endpoint(tenant_id: str, db: Session = Depends(get_db)):
    """Obtiene las métricas de consumo actuales de un tenant."""
    supported = get_supported_tenants(db)
    if tenant_id not in supported:
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' no válido.")
    return get_tenant_metrics(tenant_id)


@router.put("/tenants/{tenant_id}/config")
def update_tenant_config_endpoint(
    tenant_id: str, config_update: TenantConfigUpdate, db: Session = Depends(get_db)
):
    """Actualiza parcialmente la configuración de un tenant."""
    supported = get_supported_tenants(db)
    if tenant_id not in supported:
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' no válido.")
    try:
        updated_config = update_tenant_config(
            db, tenant_id, config_update.model_dump(exclude_unset=True)
        )
        return {"message": "Configuración actualizada", "config": updated_config}
    except Exception as e:
        logger.error(f"Error actualizando configuración de {tenant_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error al actualizar la configuración."
        )


@router.post("/tenants/{tenant_id}/upload")
async def upload_document(tenant_id: str, file: UploadFile = File(...)):
    """Endpoint para subir un archivo para un tenant específico."""
    supported = get_supported_tenants()
    if tenant_id not in supported:
        raise HTTPException(status_code=400, detail=f"Tenant '{tenant_id}' no válido.")
    if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Solo archivos .txt y .pdf")
    try:
        file_bytes = await file.read()
        upload_single_file_to_s3(tenant_id, file.filename, file_bytes)
        return {"message": f"Archivo {file.filename} procesado para {tenant_id}"}
    except Exception as e:
        logger.error(f"Error subiendo archivo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error procesando el archivo.")


@router.post("/ask")
async def ask_question(request: AskRequest):
    """Endpoint principal para realizar preguntas al sistema RAG."""
    supported = get_supported_tenants()
    if request.tenant_id not in supported:
        raise HTTPException(
            status_code=400, detail=f"Tenant '{request.tenant_id}' no válido."
        )
    try:
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
        raise HTTPException(status_code=429, detail=str(ve))
    except Exception as e:
        logger.error(f"Error procesando consulta RAG: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno en consulta RAG.")


@router.post("/evaluation/run")
def run_dynamic_evaluation(request: EvaluationRequest):
    """Ejecuta el marco de evaluación dinámica."""
    if not request.dataset:
        raise HTTPException(status_code=400, detail="Dataset vacío.")
    try:
        dataset_dicts = [item.model_dump() for item in request.dataset]
        results = run_evaluation(dataset_dicts)
        return {"message": "Evaluación completada", "results": results}
    except Exception as e:
        logger.error(f"Error en evaluación: {str(e)}")
        raise HTTPException(status_code=500, detail="Error durante la evaluación.")


@router.get("/evaluation/results")
def get_evaluation_results(db: Session = Depends(get_db)):
    """Obtiene el historial de resultados de evaluación."""
    results = (
        db.query(EvaluationResult).order_by(EvaluationResult.timestamp.desc()).all()
    )
    return {"history": results}
