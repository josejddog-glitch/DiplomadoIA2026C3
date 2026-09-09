import os
import json
from pathlib import Path
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DeepLake

from embeddings_provider import build_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> Dict:
    """Carga el archivo JSON de configuración."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_deeplake_db(cfg: Dict) -> None:
    """Crea (o sobreescribe) la base tensorial Deep Lake a partir de un PDF."""
    pdf_path = Path(cfg["pdf_path"])
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No existe el PDF indicado: {pdf_path}")

    dl_cfg = cfg["deeplake"]
    dataset_path = Path(dl_cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    overwrite = bool(dl_cfg.get("overwrite", False))

    emb_cfg = cfg["embedding"]

    split_cfg = cfg["splitter"]
    chunk_size = int(split_cfg["chunk_size"])
    chunk_overlap = int(split_cfg["chunk_overlap"])
    separators = split_cfg.get("separators", ["\n\n", "\n", ".", " ", ""])

    print(f"Cargando PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Total de páginas cargadas: {len(docs)}")

    print(f"Dividiendo en fragmentos (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    split_docs = splitter.split_documents(docs)
    print(f"Total de fragmentos generados: {len(split_docs)}")

    print(f"Usando modelo de embeddings: {emb_cfg['model_name']} (via {emb_cfg['api_base']})")
    embeddings = build_embeddings(emb_cfg["model_name"], emb_cfg["api_base"])

    print(f"Creando base Deep Lake en: {dataset_path} (overwrite={overwrite})")
    DeepLake.from_documents(
        documents=split_docs,
        embedding=embeddings,
        dataset_path=dataset_path,
        overwrite=overwrite
    )

    print("Base Deep Lake creada correctamente.")


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "config.json"
    cfg = load_config(config_file)
    build_deeplake_db(cfg)
