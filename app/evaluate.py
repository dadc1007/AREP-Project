import json
import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from app.tenants import get_supported_tenants, init_db
from app.rag import run_rag_pipeline, get_llm
from app.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)
load_dotenv()

DATASET_PATH = "app/evaluation_dataset.json"


def load_dataset():
    """
    Carga el conjunto de datos de evaluación desde un archivo JSON.

    Returns:
        list: Lista de objetos de prueba con pregunta y respuesta esperada.
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_response(
    question: str, expected_answer: str, generated_answer: str, context_sources: list
) -> dict:
    """
    Usa el enfoque 'LLM-as-a-judge' para evaluar la calidad de una respuesta generada.
    Califica la Relevancia y la Fidelidad (Faithfulness) en una escala de 1 a 5.

    Args:
        question (str): La pregunta realizada.
        expected_answer (str): La respuesta ideal (Ground Truth).
        generated_answer (str): La respuesta producida por el sistema RAG.
        context_sources (list): Lista de fuentes utilizadas para la generación.

    Returns:
        dict: Un objeto con los puntajes y feedback del evaluador.
    """
    llm = get_llm(temperature=0.0)

    prompt = PromptTemplate.from_template(
        """Eres un evaluador imparcial. Tu tarea es evaluar la respuesta de un sistema RAG en dos métricas:
1. Relevancia (1-5): ¿Qué tan bien responde a la pregunta basándose en la respuesta esperada?
2. Fidelidad/Faithfulness (1-5): ¿Parece que la respuesta está basada estrictamente en información recuperada o el modelo alucinó detalles no justificados? (5 = muy fiel, 1 = alucinación total).

Pregunta original: {question}
Respuesta esperada (Ground Truth): {expected_answer}
Respuesta generada por RAG: {generated_answer}
Fuentes citadas: {sources}

Debes responder ÚNICAMENTE con un JSON válido usando este formato:
{{
    "relevance_score": int,
    "faithfulness_score": int,
    "feedback": "Breve explicación de los puntajes"
}}
"""
    )

    try:
        response = llm.invoke(
            prompt.format(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                sources=", ".join(context_sources) if context_sources else "Ninguna",
            )
        )

        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Error en evaluación LLM: {e}")
        return {
            "relevance_score": 0,
            "faithfulness_score": 0,
            "feedback": "Error en evaluación",
        }


def run_evaluation():
    """
    Ejecuta el proceso de evaluación completo para todos los inquilinos soportados.
    Itera sobre el dataset, genera respuestas y las califica, guardando los resultados
    consolidados en un archivo JSON para su posterior análisis.
    """
    logger.info("Iniciando Marco de Evaluación Multimodelo...")
    init_db()
    dataset = load_dataset()
    results = {}
    supported_tenants = get_supported_tenants()

    for tenant_id in supported_tenants:
        logger.info(f"--- Evaluando perfil: {tenant_id} ---")
        tenant_results = {
            "total_relevance": 0,
            "total_faithfulness": 0,
            "total_time_ms": 0,
            "queries": 0,
            "details": [],
        }

        for item in dataset:
            question = item["question"]
            expected = item["expected_answer"]

            start_time = time.time()
            try:
                rag_result = run_rag_pipeline(tenant_id, question)
                generated_answer = rag_result["answer"]
                sources = rag_result["sources"]
            except Exception as e:
                logger.error(f"Error ejecutando RAG para {tenant_id}: {e}")
                generated_answer = "Error de ejecución"
                sources = []
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000
            eval_metrics = evaluate_response(
                question, expected, generated_answer, sources
            )

            tenant_results["total_relevance"] += eval_metrics.get("relevance_score", 0)
            tenant_results["total_faithfulness"] += eval_metrics.get(
                "faithfulness_score", 0
            )
            tenant_results["total_time_ms"] += elapsed_ms
            tenant_results["queries"] += 1

            tenant_results["details"].append(
                {
                    "question": question,
                    "relevance": eval_metrics.get("relevance_score", 0),
                    "faithfulness": eval_metrics.get("faithfulness_score", 0),
                    "time_ms": round(elapsed_ms, 2),
                }
            )

        # Calcular promedios
        q_count = tenant_results["queries"]

        if q_count > 0:
            tenant_results["avg_relevance"] = round(
                tenant_results["total_relevance"] / q_count, 2
            )
            tenant_results["avg_faithfulness"] = round(
                tenant_results["total_faithfulness"] / q_count, 2
            )
            tenant_results["avg_time_ms"] = round(
                tenant_results["total_time_ms"] / q_count, 2
            )

        results[tenant_id] = tenant_results

        logger.info(
            f"Resultados {tenant_id}: Relevancia Media={tenant_results.get('avg_relevance')}, Fidelidad Media={tenant_results.get('avg_faithfulness')}, Tiempo Medio={tenant_results.get('avg_time_ms')}ms"
        )

    os.makedirs("tmp_data", exist_ok=True)
    with open("tmp_data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info(
        "Evaluación completada. Resultados guardados en tmp_data/evaluation_results.json"
    )


if __name__ == "__main__":
    run_evaluation()
