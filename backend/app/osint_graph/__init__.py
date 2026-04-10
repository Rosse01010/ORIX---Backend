"""
ORIX OSINT Graph Engine
========================
Face-based Identity Intelligence Graph Platform.

Transforms facial embeddings into identity nodes with relationships,
entity links, and multi-source intelligence fusion.

Architecture:
    FACE EMBEDDING -> PERSON NODE -> ENTITY LINKS -> SOURCES -> EVENTS

Modules:
    core/       - Graph engine, identity resolver, similarity, entity linker
    storage/    - PostgreSQL graph DB + pgvector integration
    models/     - Pydantic schemas and SQLAlchemy ORM models
    ingestion/  - Dataset and source ingestion pipelines
    api/        - FastAPI route extensions
    utils/      - Scoring and normalization utilities
"""
