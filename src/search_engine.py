import os
import json
import shutil
import tempfile
import logging
import faiss
import numpy as np
from dataclasses import asdict  # <-- ДОБАВЛЕНО для конвертации dataclass Role 2 в dict
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

# === СТРОГИЕ ИМПОРТЫ МОДУЛЕЙ ДРУГИХ РОЛЕЙ ===
from paragraph_refinement import resolve_fragments, load_chunks_db
from bm25_search import search as search_bm25
# Импортируем search от Role 2 под псевдонимом, чтобы не было конфликта имен
from faiss_search import search as search_dense_raw
# encode_query тоже лежит в faiss_search.py (она импортировала её туда)
from faiss_search import encode_query
from hybrid_search import search_hybrid_rrf

# === ЛОГИРОВАНИЕ И ПУТИ ===
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_HNSW_PATH = BASE_DIR / "indexes" / "faiss_ann" / "faiss_hnsw.index"
# Путь к маппингу ID (создается Role 2, нужно нам для HNSW)
CHAPTER_IDS_PATH = BASE_DIR / "indexes" / "faiss_flat" / "chapter_ids.json"
BM25_INDEX_DIR = BASE_DIR / "indexes" / "bm25"

# === ГЛОБАЛЬНОЕ СОСТОЯНИЕ СЕРВЕРА ===
APP_STATE = {
    "faiss_hnsw": None,
    "chapter_ids": None,  # <-- ДОБАВЛЕНО для маппинга числовых ID FAISS в строковые
    "is_ready": False
}

# === НАСТРОЙКИ QUALITY GATE ===
QUALITY_GATE_THRESHOLDS = {
    "bm25": 1.5,
    "dense": 0.4,
    "dense_ann": 0.4,
    "hybrid": 2.0,
    "refined": 2.0
}


def safe_read_index(filepath: Path) -> faiss.Index:
    """Безопасное чтение FAISS индекса (фикс бага с кириллицей/OneDrive)."""
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".index")
    os.close(tmp_fd)
    shutil.copy(str(filepath), tmp_path_str)
    index = faiss.read_index(tmp_path_str)
    os.remove(tmp_path_str)
    return index


def initialize_server_state():
    if APP_STATE["is_ready"]:
        return

    logger.info("=== ЗАПУСК ONLINE СЕРВЕРА: ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ===")

    # 1. Загрузка базы текстов чанков
    logger.info("Загрузка JSONL базы чанков в оперативную память...")
    load_chunks_db()

    # 2. Загрузка маппинга индексов (КРИТИЧЕСКИ ВАЖНО ДЛЯ HNSW)
    if CHAPTER_IDS_PATH.exists():
        with open(CHAPTER_IDS_PATH, 'r', encoding='utf-8') as f:
            APP_STATE["chapter_ids"] = json.load(f)
        logger.info(f"Загружен маппинг chapter_ids ({len(APP_STATE['chapter_ids'])} записей).")
    else:
        logger.error(f"Файл маппинга не найден: {CHAPTER_IDS_PATH}. Метод dense_ann будет работать некорректно!")

    # 3. Загрузка ANN индекса HNSW
    if FAISS_HNSW_PATH.exists():
        logger.info(f"Загрузка HNSW индекса из {FAISS_HNSW_PATH}...")
        APP_STATE["faiss_hnsw"] = safe_read_index(FAISS_HNSW_PATH)
        logger.info(f"HNSW индекс загружен. Векторов: {APP_STATE['faiss_hnsw'].ntotal}")
    else:
        logger.error(f"HNSW индекс не найден по пути {FAISS_HNSW_PATH}!")

    APP_STATE["is_ready"] = True
    logger.info("=== СОСТОЯНИЕ СЕРВЕРА УСПЕШНО ИНИЦИАЛИЗИРОВАНО ===")


def _format_result(item: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """Приводит ответ к единому контракту."""
    return {
        "book": item.get("book", "Unknown Book"),
        "author": item.get("author", "Unknown Author"),
        "chapter": item.get("chapter", "Unknown Chapter"),
        "chapter_id": str(item.get("chapter_id", "N/A")),
        "best_fragment": item.get("best_fragment", "Full chapter returned"),
        "paragraph_position": item.get("paragraph_position", -1),
        "search_method": method_name,
        "score_or_rank": item.get("score", item.get("rank", 0)),
        "provenance": f"Gutenberg ID: N/A | Chapter ID: {item.get('chapter_id', 'N/A')} | Para Pos: {item.get('paragraph_position', 'N/A')}"
    }


def apply_quality_gate(results: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    if not results:
        return [{"low_confidence": True, "warning": "No results found.", "search_method": method}]

    best_result = results[0]
    top_score = best_result.get("score_or_rank", 0)
    threshold = QUALITY_GATE_THRESHOLDS.get(method, 0)

    is_low_confidence = False
    warning_reason = ""

    if top_score < threshold:
        is_low_confidence = True
        warning_reason = f"Top score ({top_score}) below threshold ({threshold})."

    if method == "refined" and "[Error:" in best_result.get("best_fragment", ""):
        is_low_confidence = True
        warning_reason = "Resolution failed."

    for res in results:
        res["low_confidence"] = is_low_confidence
        if is_low_confidence:
            res["warning"] = warning_reason

    return results


def search(query: str, method: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
    if not APP_STATE["is_ready"]:
        raise RuntimeError("Сервер не инициализирован! Вызовите initialize_server_state().")

    logger.info(f"Поиск: query='{query[:30]}...' | method={method}")
    raw_results = []

    if method == "bm25":
        # Предполагаем, что Role 1 возвращает список словарей
        raw_results = search_bm25(query, top_k)

    elif method == "dense":
        # Role 2 возвращает List[DenseResult]. Конвертируем dataclass в dict через asdict()
        dense_dataclass_results = search_dense_raw(query, top_k)
        raw_results = [asdict(r) for r in dense_dataclass_results]


    elif method == "dense_ann":
        if not APP_STATE["faiss_hnwn"] or not APP_STATE["chapter_ids"]:
            raise RuntimeError("HNSW индекс или маппинг ID не загружены!")
        query_vector = encode_query(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        scores, ids = APP_STATE["faiss_hnsw"].search(query_vector, top_k)
        # ВАЖНО: Маппим числовой индекс FAISS в реальный строковый ID чанка
        raw_results = []
        # Добавлен tqdm (leave=False чтобы не засорять логи пустыми строками после завершения)
        for rank, (score, idx) in tqdm(enumerate(zip(scores[0], ids[0]), start=1),
                                       total=len(scores[0]),
                                       desc="Mapping HNSW IDs",
                                       leave=False):
            if idx != -1:
                str_chapter_id = APP_STATE["chapter_ids"][idx]
                raw_results.append({
                    "chapter_id": str_chapter_id,
                    "score": float(score),
                    "rank": rank
                })

    elif method in ["hybrid", "refined"]:
        raw_results = search_hybrid_rrf(
            query,
            top_k,
            bm25_index_dir=BM25_INDEX_DIR
        )

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    if method == "refined":
        logger.info("Извлечение текста чанков из JSONL...")
        raw_results = resolve_fragments(raw_results)

    formatted_results = [_format_result(item, method) for item in raw_results]
    gated_results = apply_quality_gate(formatted_results, method)

    return gated_results


if __name__ == "__main__":
    logger.info("=== ТЕСТОВЫЙ ЗАПУСК SEARCH_ENGINE ===")
    try:
        initialize_server_state()
        # Тестируем твой метод dense_ann, так как он сложнее всего
        test_res = search("cozy winter night", method="dense_ann", top_k=2)

        print("\n--- РЕЗУЛЬТАТ ---")
        for r in test_res:
            print(f"ID: {r['chapter_id']} | Скор: {r['score_or_rank']}")
            if r['low_confidence']:
                print(f"WARNING: {r['warning']}")
            print("-" * 30)

    except Exception as e:
        logger.error(f"ОШИБКА: {e}")