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

## Configuration

DriveIQ uses YAML-based configuration files stored in the `configs/` directory.

Current config files:
- `configs/app.yaml`
- `configs/retrieval.yaml`
- `configs/eval.yaml`

These control runtime settings such as paths, retrieval parameters, embedding model selection, and evaluation behavior.

## Logging and Artifacts

DriveIQ writes timestamped run artifacts under `artifacts/runs/` and is designed to support traceable indexing, retrieval, and evaluation workflows.

A lightweight CLI smoke test can be run with:

    PYTHONPATH=src python -m driveiq.cli

## Core Data Contracts

DriveIQ uses typed Pydantic schemas for core objects such as:
- documents
- chunks
- retrieval responses
- summarization and Q&A responses

These schemas provide clean interfaces between ingestion, retrieval, generation, and evaluation components.

## Raw File Loading

DriveIQ begins ingestion by scanning `data/raw/` recursively, identifying supported file types, and creating typed document records with metadata.

Current supported discovery types:
- `.txt`
- `.md`
- `.pdf`
- `.png`
- `.jpg`
- `.jpeg`

## Text Parsing

DriveIQ uses a dedicated text parser for `.txt` and `.md` files. The parser performs lightweight normalization and emits parser metadata that is stored alongside each document record.

## PDF Parsing

DriveIQ includes a dedicated PDF parser built with PyMuPDF. The parser extracts text page by page and stores page-level metadata for future chunking, retrieval traceability, and evaluation workflows.

## Transcript Parsing

DriveIQ includes a transcript parser for meeting-style text files. Transcript lines with timestamp and speaker structure are normalized into retrieval-friendly text and enriched with speaker and utterance metadata.

## Image Parsing

DriveIQ includes an image parser stub for `.png`, `.jpg`, and `.jpeg` files. The current implementation establishes the ingestion contract and metadata structure for future OCR or caption-based extraction.

## Normalization Layer

DriveIQ applies a lightweight normalization step after parsing to standardize whitespace, reduce formatting noise, and preserve retrieval-friendly text structure across file types.

## Processed Document Artifacts

After ingestion and normalization, DriveIQ writes processed document records to `data/processed/` as JSON artifacts. A manifest file is also produced to summarize the processed batch.

## Chunk Building

DriveIQ converts processed documents into retrieval-ready chunk records. The initial chunking strategy uses paragraph-style segmentation and preserves source traceability through chunk metadata such as source document ID, filename, and character offsets.