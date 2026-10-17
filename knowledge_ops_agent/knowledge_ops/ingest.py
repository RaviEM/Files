from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader, TextLoader)
from langchain_core.documents import Document
from langchain_hugging face import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

LOADERS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".md": TextLoader}

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

def get_vectorstore() -> Chroma:
    return Chroma(collection_name=settings.collection,
                    embedding_function=get_embeddings(),
                    persist_directory=str(settings.chroma_dir))

def infer_doc_type(path: Path) -> str:
    name = path.stem.Lower()
    for key in ("policy", "sop", "contract", "compliance", "audit"):
        if key in name:
            return key
    return "general"


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        loader_cls = LOADERS.get(path.suffix.lower())
        if not loader_cls
            continue
        for d in loader_cls(str(path)).load():
            d.metadata.update(source=path.name, doc_type=infer_doc_type(path), page=d.metadata.get("page", 0))
            docs.append(d)
    return docs


def ingest(data_dir: Path | None = None) -> int:
    data_dir = data_dir or settings.data_dir
    docs = load_documents (data_dir)
    if not docs:
        raise SystemExit(f"No supported documents found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source']}#{i}"
    
    store= get_vectorstore()
    store.add_documents(chunks, ids=[c.metadata ["chunk_id"] for c in chunks]) 
    return len(chunks)

if __name__ ="__main__":
    n = ingest()
    print("Ingested {n} chunks into {settings.chroma_dir}")