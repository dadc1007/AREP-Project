from app.database import SessionLocal
from app.models import TenantMetrics, TenantConfig
from app.logging_config import get_logger

logger = get_logger(__name__)


def check_quota(tenant_id: str):
    """
    Verifica si un inquilino ha alcanzado su límite de tokens permitido.
    Consulta la base de datos PostgreSQL para comparar 'tokens_used' contra 'max_tokens'.
    
    Args:
        tenant_id (str): ID del inquilino a validar.
        
    Raises:
        ValueError: Si el tenant no existe o si ha excedido su cuota.
    """
    db = SessionLocal()
    try:
        metrics = (
            db.query(TenantMetrics).filter(TenantMetrics.tenant_id == tenant_id).first()
        )
        config = (
            db.query(TenantConfig).filter(TenantConfig.tenant_id == tenant_id).first()
        )

        if not metrics or not config:
            raise ValueError(f"Tenant '{tenant_id}' no existe en la base de datos.")

        if metrics.tokens_used >= config.max_tokens:
            logger.warning(
                f"Tenant {tenant_id} ha excedido su cuota ({metrics.tokens_used}/{config.max_tokens} tokens)"
            )
            raise ValueError(
                f"Cuota excedida para el tenant {tenant_id}. Límite: {config.max_tokens} tokens."
            )
    finally:
        db.close()


def add_usage(tenant_id: str, tokens: int):
    """
    Registra el consumo de tokens y aumenta el contador de consultas para un inquilino.
    La actualización se realiza de forma atómica en la base de datos PostgreSQL.
    
    Args:
        tenant_id (str): ID del inquilino.
        tokens (int): Cantidad de tokens a sumar al acumulado.
    """
    db = SessionLocal()
    try:
        metrics = (
            db.query(TenantMetrics).filter(TenantMetrics.tenant_id == tenant_id).first()
        )
        if not metrics:
            metrics = TenantMetrics(tenant_id=tenant_id, tokens_used=0, queries_count=0)
            db.add(metrics)

        metrics.tokens_used += tokens
        metrics.queries_count += 1

        db.commit()
        logger.info(
            f"Métricas actualizadas para {tenant_id}: +{tokens} tokens. Total: {metrics.tokens_used}"
        )
    except Exception as e:
        logger.error(f"Error actualizando métricas para {tenant_id}: {e}")
        db.rollback()
    finally:
        db.close()
