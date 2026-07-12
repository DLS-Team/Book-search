import time
import csv
import os
import shutil
import tempfile
import numpy as np
import faiss
import logging
from pathlib import Path
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# === БАЗОВЫЕ ПУТИ (Привязаны к местоположению скрипта) ===

BASE_DIR = Path(__file__).resolve().parent.parent
ROLE2_FLAT_INDEX_PATH = BASE_DIR / "indexes" / "faiss_flat" / "flat.index"
EMBEDDINGS_PATH = BASE_DIR / "indexes" / "faiss_flat" / "embeddings.npy"
BENCHMARK_CSV = BASE_DIR / "experiments" / "benchmark_results.csv"
INDEX_DIR = BASE_DIR / "indexes" / "faiss_ann"

# === КОНФИГУРАЦИЯ БЕНЧМАРКА ===
K = 10  # Сколько топ-результатов ищем
N_QUERIES = 200  # Количество запросов для замера latency


def load_embeddings() -> np.ndarray:
    """Жестко загружает реальные эмбеддинги от Role 2. Без заглушек."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Файл эмбеддингов не найден по пути: {EMBEDDINGS_PATH}\n"
            f"Остановите скрипт и запустите пайплайн Role 2."
        )

    vectors = np.load(str(EMBEDDINGS_PATH)).astype('float32')
    logger.info(f"Успешно загружено векторов: {vectors.shape[0]}, размерность: {vectors.shape[1]}")
    return vectors


def safe_read_index(filepath: Path) -> faiss.Index:
    """
    Безопасное чтение FAISS индекса, обходящее баг с кириллицей в пути на Windows.
    Копирует файл в системную Temp, читает его, удаляет копию.
    """
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".index")
    os.close(tmp_fd)
    shutil.copy(str(filepath), tmp_path_str)

    index = faiss.read_index(tmp_path_str)
    os.remove(tmp_path_str)
    return index


def measure_build_time(index: faiss.Index, vectors: np.ndarray) -> float:
    start = time.perf_counter()
    index.add(vectors)
    return time.perf_counter() - start


def measure_load_time(filepath: Path) -> float:
    start = time.perf_counter()
    safe_read_index(filepath) # Используем нашу новую функцию
    return time.perf_counter() - start


def measure_disk_size(filepath: Path) -> float:
    return filepath.stat().st_size / (1024 * 1024)  # в МБ


def measure_search_latency(index: faiss.Index, queries: np.ndarray, k: int, desc: str = "Searching") -> dict:
    actual_k = min(k, index.ntotal)
    index.search(queries[:5], actual_k)  # Warmup

    latencies = []
    # <-- ДОБАВЛЕН TQDM ДЛЯ НАГЛЯДНОСТИ ЗАМЕРА LATENCY
    for q in tqdm(queries, desc=desc, leave=False, unit="query"):
        start = time.perf_counter()
        index.search(q.reshape(1, -1), actual_k)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies.sort()
    return {
        "p50_ms": round(latencies[int(len(latencies) * 0.50)], 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3)
    }


def calculate_recall(gt_ids: np.ndarray, test_ids: np.ndarray) -> float:
    return np.sum(gt_ids == test_ids) / gt_ids.size


def run_index_benchmark(vectors: np.ndarray, queries: np.ndarray, index_name: str, **params) -> dict:
    dim = vectors.shape[1]

    if index_name == "FAISS_Flat":
        logger.info(f"Сборка индекса: {index_name}...")
        index = faiss.IndexFlatIP(dim)
    elif index_name == "FAISS_HNSW":
        logger.info(
            f"Сборка индекса: {index_name} (M={params.get('M')}, efC={params.get('efConstruction')}). Это займет время, FAISS строит граф в фоне...")
        index = faiss.IndexHNSWFlat(dim, params.get('M', 32), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params.get('efConstruction', 200)
        index.hnsw.efSearch = params.get('efSearch', 128)
    else:
        raise ValueError(f"Unknown index: {index_name}")

    # 1. Build Time
    build_time = measure_build_time(index, vectors)
    logger.info(f"Индекс {index_name} собран за {build_time:.2f} сек.")

    # 2. Disk Size (Фикс для OneDrive: пишем в темп, переносим в финальную папку)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Убрали _tmp из названия! Это теперь финальные файлы
    target_filepath = INDEX_DIR / f"{index_name.lower()}.index"

    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".index")
    os.close(tmp_fd)
    faiss.write_index(index, tmp_path_str)
    shutil.move(tmp_path_str, str(target_filepath))
    disk_size = measure_disk_size(target_filepath)

    # 3. Load Time (Фикс для OneDrive: читаем из темп-копии)
    tmp_fd2, tmp_path_str2 = tempfile.mkstemp(suffix=".index")
    os.close(tmp_fd2)
    shutil.copy(str(target_filepath), tmp_path_str2)
    load_time = measure_load_time(Path(tmp_path_str2))
    os.remove(tmp_path_str2)  # Удаляем только временную копию для замера скорости

    # ВАЖНО: Мы больше НЕ удаляем target_filepath!
    # Он останется на диске для search_engine.py (Таска 4.4)
    logger.info(f"Финальный индекс сохранен на диск для онлайн-сервинга: {target_filepath}")

    # 4. RAM estimation
    ram_usage = disk_size * 1.1 if "HNSW" in index_name else disk_size

    # 5. Search Latency (Передаем название для tqdm)
    latency = measure_search_latency(index, queries, K, desc=f"Latency {index_name}")

    return {
        "build_time_sec": round(build_time, 2),
        "load_time_sec": round(load_time, 2),
        "index_size_mb": round(disk_size, 2),
        "ram_usage_mb": round(ram_usage, 2),
        "p50_ms": latency["p50_ms"],
        "p95_ms": latency["p95_ms"],
        "index_object": index
    }


def save_results_to_csv(results: list):
    fieldnames = ["search_mode", "build_time_sec", "load_time_sec", "index_size_mb", "ram_usage_mb", "p50_ms", "p95_ms",
                  "recall_vs_flat"]

    with open(str(BENCHMARK_CSV), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Реальные метрики сохранены в {BENCHMARK_CSV}")


if __name__ == "__main__":
    start_total = time.perf_counter()
    logger.info("=== ЗАПУСК ФИНАЛЬНОГО БЕНЧМАРКА НА РЕАЛЬНЫХ ДАННЫХ ===")

    # Шаг 1: Загрузка векторов
    db_vectors = load_embeddings()
    actual_n_queries = min(N_QUERIES, db_vectors.shape[0])
    query_vectors = db_vectors[:actual_n_queries]

    csv_rows = []

    # Шаг 2: ИСПОЛЬЗУЕМ ГОТОВЫЙ FLAT ОТ ROLE 2 КАК ЭТАЛОН
    logger.info("=== 1/3: ЗАГРУЗКА ЭТАЛОНА FAISS FLAT (от Role 2) ===")
    if not ROLE2_FLAT_INDEX_PATH.exists():
        raise FileNotFoundError(f"Эталонный Flat индекс от Role 2 не найден по пути: {ROLE2_FLAT_INDEX_PATH}")

    # Замеряем время загрузки
    flat_load_time = measure_load_time(ROLE2_FLAT_INDEX_PATH)
    flat_size_mb = measure_disk_size(ROLE2_FLAT_INDEX_PATH)

    # Загружаем в память для поиска (используем safe_read из-за кириллицы)
    flat_index = safe_read_index(ROLE2_FLAT_INDEX_PATH)
    logger.info(f"Flat индекс загружен из памяти Role 2. Векторов: {flat_index.ntotal}")
    # Замеряем latency Flat
    flat_latency = measure_search_latency(flat_index, query_vectors, K, desc="Latency FAISS_Flat")

    # Получаем Ground Truth (идеальные результаты) для расчета Recall у HNSW
    _, gt_ids = flat_index.search(query_vectors, K)

    csv_rows.append({
        "search_mode": "Dense_Flat",
        "build_time_sec": "N/A (Собран Role 2)",  # Мы его не строили
        "load_time_sec": round(flat_load_time, 2),
        "index_size_mb": round(flat_size_mb, 2),
        "ram_usage_mb": round(flat_size_mb, 2),
        "p50_ms": flat_latency["p50_ms"],
        "p95_ms": flat_latency["p95_ms"],
        "recall_vs_flat": 1.0
    })

    # Шаг 3: Бенчмарк HNSW (Сбалансированный)
    logger.info("=== 2/3: СБОРКА И ТЕСТИРОВАНИЕ FAISS HNSW (Сбалансированный) ===")
    hnsw_metrics = run_index_benchmark(db_vectors, query_vectors, "FAISS_HNSW", M=32, efConstruction=200, efSearch=128)
    _, hnsw_ids = hnsw_metrics.pop("index_object").search(query_vectors, K)

    csv_rows.append({
        "search_mode": "Dense_ANN_HNSW",
        "recall_vs_flat": round(calculate_recall(gt_ids, hnsw_ids), 4),
        **hnsw_metrics
    })

    # Шаг 4: Бенчмарк HNSW (Агрессивный)
    logger.info("=== 3/3: СБОРКА И ТЕСТИРОВАНИЕ FAISS HNSW (Агрессивный/Быстрый) ===")
    hnsw_fast_metrics = run_index_benchmark(db_vectors, query_vectors, "FAISS_HNSW", M=16, efConstruction=100,
                                            efSearch=32)
    _, hnsw_fast_ids = hnsw_fast_metrics.pop("index_object").search(query_vectors, K)

    csv_rows.append({
        "search_mode": "Dense_ANN_HNSW_Fast",
        "recall_vs_flat": round(calculate_recall(gt_ids, hnsw_fast_ids), 4),
        **hnsw_fast_metrics
    })

    # Шаг 5: Финализация
    save_results_to_csv(csv_rows)

    total_time = time.perf_counter() - start_total
    logger.info(f"=== БЕНЧМАРК УСПЕШНО ЗАВЕРШЕН ЗА {total_time:.2f} СЕКУНД ===")
    logger.info("HNSW индексы сохранены в indexes/faiss.ann/ для online-сервинга (Таска 4.4).")
    logger.info("Передай файл experiments/benchmark_results.csv Role 3.")