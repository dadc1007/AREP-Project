import os

TENANTS = {
    "tenant_a": {
        "collection": "db_tenant_a",
        "embedding": "miniLM",  # Simulado: la arquitectura permite cambiar modelos por tenant
    },
    "tenant_b": {"collection": "db_tenant_b", "embedding": "miniLM"},
    "tenant_c": {"collection": "db_tenant_c", "embedding": "miniLM"},
}

# Lista de tenants soportados extraída de la configuración
SUPPORTED_TENANTS = list(TENANTS.keys())


def get_tenant_db_path(tenant_id: str) -> str:
    """
    Retorna la ruta donde se guardará/leerá la base de datos vectorial
    aislada para cada tenant.
    """
    if tenant_id not in SUPPORTED_TENANTS:
        raise ValueError(f"Tenant {tenant_id} no soportado.")

    # Leer la configuración específica del tenant
    config = TENANTS[tenant_id]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, config["collection"])


def get_tenant_data_path(tenant_id: str) -> str:
    """
    Retorna la ruta donde están los documentos crudos de cada tenant.
    """
    if tenant_id not in SUPPORTED_TENANTS:
        raise ValueError(f"Tenant {tenant_id} no soportado.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", tenant_id)
