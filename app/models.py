from sqlalchemy import Column, String, Integer, Boolean, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base


class TenantConfig(Base):
    __tablename__ = "tenant_configs"

    tenant_id = Column(String, primary_key=True, index=True)
    namespace = Column(String, unique=True, index=True)
    chunk_size = Column(Integer, default=500)
    chunk_overlap = Column(Integer, default=50)
    use_hyde = Column(Boolean, default=False)
    use_reranking = Column(Boolean, default=False)
    temperature = Column(Float, default=0.0)
    system_prompt = Column(String)
    max_tokens = Column(Integer, default=5000)


class TenantMetrics(Base):
    __tablename__ = "tenant_metrics"

    tenant_id = Column(String, primary_key=True, index=True)
    tokens_used = Column(Integer, default=0)
    queries_count = Column(Integer, default=0)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String, index=True)
    avg_relevance = Column(Float)
    avg_faithfulness = Column(Float)
    avg_time_ms = Column(Float)
    timestamp = Column(DateTime, server_default=func.now())
