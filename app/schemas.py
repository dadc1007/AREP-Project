from pydantic import BaseModel
from typing import Optional, List


class AskRequest(BaseModel):
    """Esquema para una consulta RAG."""

    question: str
    tenant_id: str


class TenantConfigUpdate(BaseModel):
    """Esquema para actualizar la configuración de un tenant."""

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    use_hyde: Optional[bool] = None
    use_reranking: Optional[bool] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None


class EvaluationItem(BaseModel):
    """Esquema para un ítem de evaluación dinámica."""

    question: str
    expected_answer: str


class EvaluationRequest(BaseModel):
    """Esquema para una petición de evaluación dinámica completa."""

    dataset: List[EvaluationItem]
