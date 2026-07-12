import logging
import time
from typing import List, Dict, Any
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Имя модели Cross-Encoder, оптимальное для скорости и качества
# Использует архитектуру схожую с bi-encoder, но работает за O(N) на коротких текстах
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SNIPPET_LENGTH_TOKENS = 512  # Стандартный лимит для MiniLM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cross_encoder():
    """Ленивая загрузка тяжелой модели (выполняется один раз)."""
    try:
        logger.info(f"Загрузка Cross-Encoder модели: {CROSS_ENCODER_MODEL_NAME}...")
        model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=DEVICE)
        logger.info("Модель загружена.")
        return model
    except ImportError:
        raise ImportError(
            "Библиотека sentence-transformers не установлена. "
            "Cross-encoder rerank недоступен. Установите: pip install sentence-transformers"
        )


def rerank_snippets(query: str, snippets: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
    """
    Переранжирует короткие сниппеты с помощью Cross-Encoder.

    :param query: Оригинальный запрос
    :param snippets: Список результатов после paragraph_refinement (должны содержать 'best_fragment')
    :param top_k: Сколько лучших результатов вернуть
    :return: Переранжированный список
    """
    model = load_cross_encoder()

    # Подготовка пар (запрос, фрагмент) для модели
    pairs = []
    valid_snippets = []

    for item in snippets:
        fragment = item.get("best_fragment", "")
        # Базовая проверка на лимит длины (в реальности токенизатор обрежет, но мы логируем)
        if len(fragment.split()) > MAX_SNIPPET_LENGTH_TOKENS:
            logger.warning(f"Сниппет превышает лимит в {MAX_SNIPPET_LENGTH_TOKENS} токенов. Возможна потеря контекста.")

        pairs.append((query, fragment))
        valid_snippets.append(item)

    if not pairs:
        return snippets

    # Предсказание скоров (этап, который занимает время)
    scores = model.predict(pairs)

    # Связываем скоры с исходными объектами и сортируем
    scored_snippets = list(zip(valid_snippets, scores))
    scored_snippets.sort(key=lambda x: x[1], reverse=True)

    # Добавляем cross-encoder score в метаданные для прозрачности
    final_results = []
    for item, score in scored_snippets[:top_k]:
        item["rerank_score"] = round(float(score), 4)
        item["search_method"] = item.get("search_method", "Unknown") + " + Reranked"
        final_results.append(item)

    return final_results


def measure_rerank_latency(query: str, snippets: List[Dict[str, Any]], runs: int = 10) -> float:
    """Замеряет среднее время работы rerank на заданном числе запросов."""
    # Warmup (первый вызов всегда долгий из-за инициализации графов PyTorch)
    rerank_snippets(query, snippets)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        rerank_snippets(query, snippets)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return round(sum(latencies) / len(latencies), 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Эмуляция входящих данных от таски 4.2 (Paragraph Refinement)
    mock_refined_snippets = [
        {"chapter_id": "ch_042",
         "best_fragment": "It was a cozy winter night near a fireplace, with a blanket wrapped around her shoulders."},
        {"chapter_id": "ch_001", "best_fragment": "He stopped under a flickering lamppost, looking for shelter."},
        {"chapter_id": "ch_105", "best_fragment": "He couldn't sleep, staring at the ceiling all night."}
    ]

    test_query = "cozy winter night near a fireplace"

    logger.info(f"Запуск Cross-Encoder Rerank для запроса: '{test_query}'")

    try:
        # 1. Демонстрация работы
        reranked = rerank_snippets(test_query, mock_refined_snippets, top_k=1)

        print("\n--- РЕЗУЛЬТАТ RERANK ---")
        best = reranked[0]
        print(f"Лучшая глава: {best['chapter_id']}")
        print(f"Cross-Encoder Score: {best['rerank_score']}")
        print(f"Фрагмент: {best['best_fragment']}")

        # 2. Замер latency (включает работу GPU/CPU нейросети)
        avg_latency = measure_rerank_latency(test_query, mock_refined_snippets)
        logger.info(f"Добавленная latency Cross-Encoder (на 3 сниппетах, без учета загрузки модели): {avg_latency} ms")

    except ImportError as e:
        logger.error(e)