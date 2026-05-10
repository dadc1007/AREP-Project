from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import TenantConfig, TenantMetrics
import logging

logger = logging.getLogger(__name__)

DEFAULT_TENANTS = {
    "tenant_a": {
        "namespace": "tenant_a",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "use_hyde": False,
        "use_reranking": False,
        "temperature": 0.0,
        "system_prompt": "Eres un asistente corporativo de tenant_a. Responde de manera concisa y directa. Si la respuesta exacta no está en el texto, usa tu conocimiento general, pero aclara que es una sugerencia tuya y no una política oficial de la empresa.",
        "max_tokens": 5000,
    },
    "tenant_b": {
        "namespace": "tenant_b",
        "chunk_size": 300,
        "chunk_overlap": 30,
        "use_hyde": True,
        "use_reranking": True,
        "temperature": 0.3,
        "system_prompt": "Eres un analista detallista de tenant_b. Analiza la información exhaustivamente. Si la respuesta exacta no está en el texto, usa tu conocimiento general, pero aclara que es una sugerencia tuya y no una política oficial de la empresa.",
        "max_tokens": 10000,
    },
    "tenant_c": {
        "namespace": "tenant_c",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "use_hyde": False,
        "use_reranking": True,
        "temperature": 0.1,
        "system_prompt": "Eres un asesor legal de tenant_c. Sé muy formal y estricto. Si la respuesta exacta no está en el texto, usa tu conocimiento general, pero aclara que es una sugerencia tuya y no una política oficial de la empresa.",
        "max_tokens": 2000,
    },
}


def init_db():
    """
    Crea las tablas en la base de datos si no existen y popula los datos
    de los inquilinos por defecto (tenant_a, tenant_b, tenant_c).
    """
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(TenantConfig).first() is None:
            logger.info("Poblando base de datos con tenants por defecto...")
            for t_id, t_data in DEFAULT_TENANTS.items():
                config = TenantConfig(
                    tenant_id=t_id,
                    namespace=t_data["namespace"],
                    chunk_size=t_data["chunk_size"],
                    chunk_overlap=t_data["chunk_overlap"],
                    use_hyde=t_data["use_hyde"],
                    use_reranking=t_data["use_reranking"],
                    temperature=t_data["temperature"],
                    system_prompt=t_data["system_prompt"],
                    max_tokens=t_data["max_tokens"],
                )
                metrics = TenantMetrics(tenant_id=t_id, tokens_used=0, queries_count=0)
                db.add(config)
                db.add(metrics)
            db.commit()
            logger.info("Base de datos inicializada correctamente.")
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        db.rollback()
    finally:
        db.close()


def get_supported_tenants(db: Session = None):
    """
    Consulta la base de datos para obtener la lista de todos los IDs de inquilinos registrados.

    Args:
        db (Session, optional): Sesión de base de datos existente. Si no se provee, se crea una nueva.

    Returns:
        list: Lista de strings con los identificadores de los tenants.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        tenants = db.query(TenantConfig.tenant_id).all()
        return [t[0] for t in tenants]
    finally:
        if close_db:
            db.close()


def get_tenant_config_dict(db: Session, tenant_id: str):
    """
    Obtiene la configuración completa de un inquilino y la retorna como un diccionario plano.

    Args:
        db (Session): Sesión de la base de datos.
        tenant_id (str): ID del inquilino.

    Returns:
        dict: Diccionario con campos como chunk_size, use_hyde, system_prompt, etc.
    """
    config = db.query(TenantConfig).filter(TenantConfig.tenant_id == tenant_id).first()
    if not config:
        return {}
    return {
        "tenant_id": config.tenant_id,
        "namespace": config.namespace,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "use_hyde": config.use_hyde,
        "use_reranking": config.use_reranking,
        "temperature": config.temperature,
        "system_prompt": config.system_prompt,
        "max_tokens": config.max_tokens,
    }


def update_tenant_config(db: Session, tenant_id: str, update_data: dict) -> dict:
    """
    Actualiza los parámetros de configuración de un inquilino de forma parcial.

    Args:
        db (Session): Sesión de la base de datos.
        tenant_id (str): ID del inquilino a actualizar.
        update_data (dict): Diccionario con los campos y nuevos valores.

    Returns:
        dict: El diccionario de configuración actualizado.
    """
    config = db.query(TenantConfig).filter(TenantConfig.tenant_id == tenant_id).first()
    if not config:
        raise ValueError(f"Tenant '{tenant_id}' no encontrado.")

    # Filtrar solo los valores que no son None
    update_fields = {k: v for k, v in update_data.items() if v is not None}

    for key, value in update_fields.items():
        if hasattr(config, key):
            setattr(config, key, value)

    db.commit()
    db.refresh(config)

    return get_tenant_config_dict(db, tenant_id)
