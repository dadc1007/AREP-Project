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
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Verificar si ya existen
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
    # Permite inyectar db, si no crea una sesion nueva rápida
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
