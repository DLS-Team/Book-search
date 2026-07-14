import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# === КОНТРАКТ С ROLE 1 ===
BASE_DIR = Path(__file__).resolve().parent.parent
CHAPTERS_DATA_PATH = BASE_DIR / "data" / "processed" / "processed_chapters.jsonl"

# Кэш базы данных в оперативной памяти
_CHUNK_DB = None


def load_chunks_db() -> Dict[str, Dict[str, Any]]:
    """
    Загружает JSONL файл в память ОДИН раз.
    Возвращает словарь вида: { "chapter_id": { "text": "...", "title": "...", "author": "..." } }
    """
    global _CHUNK_DB

    if _CHUNK_DB is not None:
        return _CHUNK_DB

    if not CHAPTERS_DATA_PATH.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {CHAPTERS_DATA_PATH}")

    logger.info(f"Загрузка датасета чанков из {CHAPTERS_DATA_PATH} в оперативную память...")
    _CHUNK_DB = {}

    # Читаем файл построчно (формат JSONL)
    with open(CHAPTERS_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            chunk_data = json.loads(line.strip())
            chunk_id = chunk_data.get("chapter_id")
            if chunk_id:
                _CHUNK_DB[chunk_id] = chunk_data

    logger.info(f"Успешно загружено {len(_CHUNK_DB)} чанков в память.")
    return _CHUNK_DB


def resolve_fragments(top_chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Таска 4.2 (Новая реальность): Чанк = Параграф.
    Базовый поиск уже вернул нужные чанки. Нам нужно только "разрешить" (resolve)
    их ID в реальный текст и метаданные для выдачи пользователю.
    """
    db = load_chunks_db()

    resolved_results = []

    for original_result in top_chapters:
        result = dict(original_result)
        ch_id = str(result.get("chapter_id"))

        if ch_id not in db:
            logger.warning(f"Чанк {ch_id} не найден в JSONL. Пропускаем.")
            result["best_fragment"] = "[Error: Text not found in dataset]"
            result["paragraph_position"] = -1
            resolved_results.append(result)
            continue

        # Достаем данные из базы
        chunk_data = db[ch_id]

        # Заполняем схему результата (раздел 2.3 плана)
        # Текст чанка становится итоговым фрагментом для пользователя
        result["best_fragment"] = chunk_data.get("text", "")

        # Позиция абзаца по сути равна ID чанка, но для схемы оставляем 0
        # (так как чанк неразделим в нашей парадигме)
        result["paragraph_position"] = 0

        # Дополняем метаданными от Role 1, если их не было в ответе базового поиска
        result["book"] = result.get("book", chunk_data.get("title", "Unknown Title"))
        result["author"] = result.get("author", chunk_data.get("author", "Unknown Author"))
        result["chapter"] = result.get("chapter", chunk_data.get("chapter_title", ch_id))

        resolved_results.append(result)

    return resolved_results


if __name__ == "__main__":
    # Эмуляция того, что вернул базовый поиск (например, FAISS) - только ID и скоры
    mock_search_engine_output = [
        {"chapter_id": "41496-8_chunk_000001", "score": 0.95},
        {"chapter_id": "some_non_existent_chunk", "score": 0.80}  # Тест ошибки
    ]

    test_query = "Addison life dramatic incident"
    logger.info(f"Тестовый запрос: '{test_query}'")

    # Вызываем резолвер
    resolved_output = resolve_fragments(mock_search_engine_output)

    print("\n--- РЕЗУЛЬТАТ RESOLVE (ТАСКА 4.2) ---")
    for res in resolved_output:
        print(f"Книга: {res['book']} | Автор: {res['author']}")
        print(f"Фрагмент (первые 150 символов): {res['best_fragment'][:150]}...")
        print("-" * 40)