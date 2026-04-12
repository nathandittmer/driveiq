from __future__ import annotations

from driveiq.schemas.response import RetrievedChunk


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No supporting context was retrieved."

    formatted_blocks: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        source_filename = chunk.metadata.get("source_filename", "unknown")
        source_type = chunk.metadata.get("source_type", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", "unknown")

        formatted_blocks.append(
            "\n".join(
                [
                    f"[Chunk {idx}]",
                    f"Source file: {source_filename}",
                    f"Source type: {source_type}",
                    f"Chunk index: {chunk_index}",
                    f"Content: {chunk.text}",
                ]
            )
        )

    return "\n\n".join(formatted_blocks)


def build_summary_prompt(document_text: str, document_title: str | None = None) -> str:
    title_line = f"Document title: {document_title}\n" if document_title else ""

    return (
        "You are helping a productivity user understand a document quickly.\n"
        "Write a concise, accurate summary of the document.\n"
        "Focus on the main purpose, key points, and any next steps or risks.\n"
        "Do not invent facts.\n\n"
        f"{title_line}"
        f"Document content:\n{document_text}\n\n"
        "Return a short paragraph summary."
    )


def build_grounded_qa_prompt(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_block = format_retrieved_chunks(retrieved_chunks)

    return (
        "You are answering a user question using only the provided retrieved context.\n"
        "If the answer is not supported by the context, say that the context does not contain enough information.\n"
        "Be concise and accurate.\n\n"
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Return a short answer grounded in the context."
    )


def build_cross_document_brief_prompt(
    user_goal: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    context_block = format_retrieved_chunks(retrieved_chunks)

    return (
        "You are helping a user synthesize information across multiple documents.\n"
        "Create a concise brief that combines the most relevant points from the provided context.\n"
        "Highlight major themes, risks, and next steps when present.\n"
        "Do not invent facts beyond the provided context.\n\n"
        f"User goal:\n{user_goal}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Return a short multi-bullet brief."
    )


def build_action_items_prompt(retrieved_chunks: list[RetrievedChunk]) -> str:
    context_block = format_retrieved_chunks(retrieved_chunks)

    return (
        "You are extracting action items from retrieved work documents.\n"
        "Only include action items that are directly supported by the context.\n"
        "If no clear action items exist, say that none were identified.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Return a short bullet list of action items."
    )