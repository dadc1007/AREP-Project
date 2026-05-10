from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.tenants import init_db
from app.logging_config import setup_logging, get_logger
from app.api import router as api_router

# Configurar logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Multitenant RAG API",
    description="Proyecto académico de arquitectura RAG con soporte multitenante.",
    version="1.1.0",
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
    """Inicializa la base de datos y los tenants por defecto al arrancar."""
    logger.info("Inicializando base de datos PostgreSQL...")
    init_db()


# Incluir las rutas de la API
app.include_router(api_router)


@app.get("/")
def read_root():
    """Endpoint de salud de la API."""
    return {"message": "Multitenant RAG API corriendo correctamente (Refactorizada)."}
