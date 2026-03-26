# DriveIQ

## Overview
DriveIQ is a production-style multimodal retrieval and summarization system inspired by Google Drive and Workspace AI features.

The system ingests documents (PDFs, text, transcripts, images), builds a retrieval index, and enables grounded summarization and question answering.

## Architecture (High-Level)
- Python ML application layer (ingestion, chunking, embeddings, generation)
- C++ retrieval core (similarity scoring and ranking)
- FastAPI service layer
- Evaluation and quality pipeline

## Status
Project initialized. Core components under development.

## Local Setup

### Create virtual environment
python -m venv .venv