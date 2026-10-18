"""Build the vector index: PDF -> clean -> chunk -> embed -> ChromaDB (with citation metadata)."""
import argparse
import json
import os
from collections import Counter

import chromadb

from hindi_rag import COLLECTION, DEFAULT_PDF, Embedder, build_chunks, extract_pages, force_utf8_output


def main() -> None:
    force_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--db", default="chroma_db", help="ChromaDB persistence directory")
    ap.add_argument("--max-chars", type=int, default=450, help="max characters per chunk")
    ap.add_argument("--overlap", type=int, default=1, help="sentences repeated between chunks")
    ap.add_argument("--tables", choices=["pymupdf", "docling"], default="pymupdf",
                    help="who reads the tables; body text is always PyMuPDF (docling needs: pip install docling)")
    ap.add_argument("--chunks-out", default="chunks.jsonl", help="where to write the readable chunk dump")
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL"),
                    help="sentence-transformers model (default BAAI/bge-m3)")
    args = ap.parse_args()

    pages = extract_pages(args.pdf, tables=args.tables)
    chunks = build_chunks(pages, args.max_chars, args.overlap)

    emb = Embedder(args.model)
    vectors = emb.encode_docs([c.embed_text for c in chunks])

    client = chromadb.PersistentClient(path=args.db)
    try:
        client.delete_collection(COLLECTION)  # re-running ingest rebuilds the index
    except Exception:
        pass
    coll = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": emb.name})
    coll.add(
        ids=[str(c.id) for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors.tolist(),
        metadatas=[{"chunk_id": c.id, "page": c.page, "section_no": c.section_no,
                    "section": c.section, "subsection": c.subsection, "kind": c.kind} for c in chunks],
    )

    with open(args.chunks_out, "w", encoding="utf-8") as f:  # human-readable dump for inspection
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    kinds = Counter(c.kind for c in chunks)
    lens = [len(c.text) for c in chunks]
    print(f"pages indexed : {len(pages)}")
    print(f"chunks        : {len(chunks)}  {dict(kinds)}")
    print(f"chunk length  : min {min(lens)} / avg {sum(lens) // len(lens)} / max {max(lens)} chars")
    print(f"embed model   : {emb.name}")
    print(f"vector db     : ChromaDB at ./{args.db}  (collection '{COLLECTION}')")
    print(f"sub-headings  : {len({(c.page, c.subsection) for c in chunks if c.subsection})} (stored as metadata, not body text)")
    print(f"table lead-ins: {sum(1 for p in pages for i in p.table_intros if i)} kept with their tables")
    print(f"chunk dump    : {args.chunks_out}")


if __name__ == "__main__":
    main()