import os

TENANTS = {
    "tenant_a": {"namespace": "tenant_a"},
    "tenant_b": {"namespace": "tenant_b"},
    "tenant_c": {"namespace": "tenant_c"},
}

# Lista de tenants soportados extraída de la configuración
SUPPORTED_TENANTS = list(TENANTS.keys())
