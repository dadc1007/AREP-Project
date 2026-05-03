import logging
import sys


def setup_logging():
    """
    Configura el sistema de logging para la aplicación.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str):
    """
    Retorna una instancia de logger para el módulo dado.
    """
    return logging.getLogger(name)
