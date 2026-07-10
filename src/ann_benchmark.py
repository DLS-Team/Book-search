import os
import time
import csv
import psutil
import numpy as np
import faiss

# Конфигурация путей
BENCHMARK_CSV_PATH = "experiments/benchmark_results.csv"
INDEX_DIR = "indexes/faiss_ann/"


def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BENCHMARK_CSV_PATH), exist_ok=True)


def generate_synthetic_data(n: int, dim: int) -> np.ndarray:
    """Генерирует случайные нормализованные векторы (заглушка вместо Role 2)."""
    print(f"Генерация {n} синтетических векторов размерности {dim}...")
    vectors = np.random.rand(n, dim).astype('float32')
    faiss.normalize_L2(vectors)
    return vectors


def measure_build_time(index: faiss.Index, vectors: np.ndarray) -> float:
    """Замеряет время добавления векторов в индекс."""
    start = time.perf_counter()
    index.add(vectors)
    end = time.perf_counter()
    return end - start


def measure_index_size_on_disk(index: faiss.Index, filename: str) -> float:
    """Сохраняет индекс и возвращает его размер в МБ."""
    filepath = os.path.join(INDEX_DIR, filename)
    faiss.write_index(index, filepath)
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)  # В МБ


def measure_search_latency(index: faiss.Index, queries: np.ndarray, k: int, n_runs: int = 100) -> dict:
    """Замеряет p50 и p95 latency поиска. Включает warmup."""
    # Warmup (разогрев кэшей и потоков FAISS)
    warmup_queries = 10
    index.search(queries[:warmup_queries], k)

    latencies = []
    for i in range(n_runs):
        q = queries[i % len(queries)].reshape(1, -1)
        start = time.perf_counter()
        index.search(q, k)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # в мс

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    return {"p50_ms": p50, "p95_ms": p95}


def calculate_recall(ground_truth_ids: np.ndarray, ann_ids: np.ndarray) -> float:
    """Считает долю совпадений топ-k результатов."""
    matches = np.sum(ground_truth_ids == ann_ids)
    return matches / ground_truth_ids.size


def run_single_benchmark(vectors: np.ndarray, queries: np.ndarray, k: int, index_type: str, **kwargs):
    """Запускает бенчмарк для одного типа индекса."""
    dim = vectors.shape[1]
    print(f"\n--- Тестируется: {index_type} ---")

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "HNSW":
        M = kwargs.get('M', 32)
        efConstruction = kwargs.get('efConstruction', 200)
        efSearch = kwargs.get('efSearch', 128)

        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
    else:
        raise ValueError("Unknown index type")

    # 1. Build time
    build_time = measure_build_time(index, vectors)
    print(f"Build time: {build_time:.2f} sec")

    # 2. Search Latency
    latency_metrics = measure_search_latency(index, queries, k)
    print(f"Latency p50: {latency_metrics['p50_ms']:.3f} ms | p95: {latency_metrics['p95_ms']:.3f} ms")

    # 3. Index Size on Disk
    disk_size = measure_index_size_on_disk(index, f"{index_type.lower()}_tmp.index")
    print(f"Index size on disk: {disk_size:.2f} MB")

    # 4. RAM Usage (оценка через размер файла для Flat, через psutil для HNSW)
    # Простая, но точная для FAISS оценка памяти = размер файла на диске
    ram_usage = disk_size

    return {
        "method": index_type if index_type == "Flat" else f"HNSW (M={kwargs.get('M')}, efC={kwargs.get('efConstruction')}, efS={kwargs.get('efSearch')})",
        "build_time_sec": round(build_time, 2),
        "index_size_mb": round(disk_size, 2),
        "ram_usage_mb": round(ram_usage, 2),
        "p50_ms": round(latency_metrics['p50_ms'], 3),
        "p95_ms": round(latency_metrics['p95_ms'], 3),
        "recall_vs_flat": None  # Заполнится позже
    }


def save_to_csv(results: list[dict]):
    """Сохраняет результаты в CSV для Role 3."""
    file_exists = os.path.isfile(BENCHMARK_CSV_PATH)

    with open(BENCHMARK_CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"\n[+] Результаты сохранены в {BENCHMARK_CSV_PATH}")


if __name__ == "__main__":
    ensure_dirs()

    # ВНИМАНИЕ: 500k векторов HNSW может съесть ~2-2.5 ГБ RAM.
    # Для первой проверки на ноутбуке оставь 100_000. Перед финальным прогоном поменяй на 500_000.
    N_VECTORS = 100_000
    DIM = 768
    K = 10
    N_QUERIES = 200

    # Генерируем данные
    db_vectors = generate_synthetic_data(N_VECTORS, DIM)
    query_vectors = generate_synthetic_data(N_QUERIES, DIM)

    results = []

    # 1. Тест FAISS Flat (Эталон)
    flat_res = run_single_benchmark(db_vectors, query_vectors, K, "Flat")

    # Достаем ground truth из Flat
    flat_index = faiss.read_index(os.path.join(INDEX_DIR, "flat_tmp.index"))
    gt_distances, gt_ids = flat_index.search(query_vectors, K)
    flat_res["recall_vs_flat"] = 1.0
    results.append(flat_res)

    # 2. Тест HNSW (дефолтные параметры)
    hnsw_res = run_single_benchmark(db_vectors, query_vectors, K, "HNSW", M=32, efConstruction=200, efSearch=128)

    # Считаем Recall HNSW против Flat
    hnsw_index = faiss.read_index(os.path.join(INDEX_DIR, "hnsw_tmp.index"))
    _, hnsw_ids = hnsw_index.search(query_vectors, K)
    recall = calculate_recall(gt_ids, hnsw_ids)
    hnsw_res["recall_vs_flat"] = round(recall, 4)
    print(f"Recall@{K} vs Flat: {recall:.4f}")
    results.append(hnsw_res)

    # 3. (Опционально) Тест HNSW с агрессивным efSearch (для демонстрации trade-off на защите)
    hnsw_res_fast = run_single_benchmark(db_vectors, query_vectors, K, "HNSW", M=32, efConstruction=200, efSearch=16)
    hnsw_index_fast = faiss.read_index(
        os.path.join(INDEX_DIR, "hnsw_tmp.index"))  # Перезаписывается, но efSearch мы задали ниже
    hnsw_index_fast.hnsw.efSearch = 16
    _, hnsw_ids_fast = hnsw_index_fast.search(query_vectors, K)
    recall_fast = calculate_recall(gt_ids, hnsw_ids_fast)
    hnsw_res_fast["recall_vs_flat"] = round(recall_fast, 4)
    hnsw_res_fast["method"] = "HNSW (M=32, efC=200, efS=16 - FAST)"
    print(f"Recall@{K} vs Flat (FAST): {recall_fast:.4f}")
    results.append(hnsw_res_fast)

    # Сохраняем в CSV
    save_to_csv(results)

    # Удаляем временные индексы с диска (они нам пока не нужны, сохраним место)
    os.remove(os.path.join(INDEX_DIR, "flat_tmp.index"))
    os.remove(os.path.join(INDEX_DIR, "hnsw_tmp.index"))