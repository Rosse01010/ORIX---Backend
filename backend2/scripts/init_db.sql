-- Executed automatically by the pgvector/pgvector Docker image on first boot.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()
