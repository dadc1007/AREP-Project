from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.tenants import SUPPORTED_TENANTS
from app.rag import run_rag_pipeline
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


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Endpoint principal para realizar preguntas al sistema RAG multitenante.
    """
    # 1. Validar que el tenant exista
    if request.tenant_id not in SUPPORTED_TENANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Tenant '{request.tenant_id}' no válido. Opciones: {SUPPORTED_TENANTS}",
        )

    try:
        # 2. Ejecutar el pipeline RAG
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
