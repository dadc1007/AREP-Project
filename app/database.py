import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")


engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    Generador de sesiones de base de datos para ser utilizado como dependencia en FastAPI.
    Asegura que la conexión se cierre después de completar la petición.

    Yields:
        Session: Sesión de SQLAlchemy activa.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
