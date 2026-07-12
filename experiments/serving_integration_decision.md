# Serving Integration Decision (Task 4.4)

## 1. Архитектурный паттерн
Для реализации единого search interface (`src/search_engine.py`) применен паттерн **Facade (Фасад)**. 
Модуль не содержит логики ранжирования (BM25, Dense, RRF). Он выступает роутером: принимает запрос, выбирает нужный модуль другой роли, получает сырые данные и приводит их к единому выходному контракту.

## 2. Строгая выходная схема (Result Schema)
Каждый результат, возвращаемый из `search_engine.py`, строго соответствует формату раздела 2.3 плана проекта:
- `book` (string)
- `author` (string)
- `chapter` (string)
- `best_fragment` (string) — заполняется текстом главы, если refinement не применялся, или конкретным абзацем, если применялся.
- `search_method` (string) — имя метода, которое пробрасывается из роутера.
- `score_or_rank` (float/int) — адаптивное поле (возвращает score для BM25/Dense и rank для Hybrid RRF).
- `provenance` (string) — собранная строка формата `Gutenberg ID | Chapter ID | Para Pos`.

## 3. Обработка зависимостей (Graceful Degradation)
Так как разработка идет параллельно, `search_engine.py` использует механизм `try/except` при импорте модулей `bm25_search`, `faiss_search`, `hybrid_search`. 
Если код другой роли еще не готов, система не падает, а подменяет его на `Mock`-объекты. Это позволяет:
1. Разрабатывать и тестировать фронтенд/бэкенд (Таска 2.7) прямо сейчас.
2. Интегрировать реальные модули без изменения кода `search_engine.py` (достаточно просто положить файл от Role 1/2/3 в папку `src/`).

## 4. Реализация каскада (Retrieve -> Rank -> Refine)
В методе `search(method="refined")` реализована логика двухэтапного пайплайна:
1. **Этап 1:** Вызов базового поиска (Hybrid RRF от Role 3) для получения топа релевантных глав.
2. **Этап 2:** Передача списка глав в `paragraph_refinement.refine_results()` (Таска 4.2). Функция обогащает словари, добавляя поля `best_fragment` и `paragraph_position`, после чего они попадают в `_format_result`.

## 5. Исключенные методы
Методы `"multi_query"` и `"reranked"` (Cross-encoder) не включены в роутер:
- `multi_query` ожидает реализации от Role 3.
- `reranked` осознанно отклонен в Таске 4.3 по причинам latency/веса.

==================================================
ТЕСТИРОВАНИЕ ЕДИНОГО SEARCH ENGINE
==================================================

--- Режим: BM25 ---
Книга: Mock Book A | Метод: bm25
Фрагмент: Full chapter returned (refinement disabled)...
Provenance: Gutenberg ID: 1001 | Chapter ID: ch_mock_1 | Para Pos: N/A
----------------------------------------
Книга: Mock Book B | Метод: bm25
Фрагмент: Full chapter returned (refinement disabled)...
Provenance: Gutenberg ID: 1002 | Chapter ID: ch_mock_2 | Para Pos: N/A
----------------------------------------

--- Режим: DENSE ---
Книга: Mock Book C | Метод: dense
Фрагмент: Full chapter returned (refinement disabled)...
Provenance: Gutenberg ID: 1003 | Chapter ID: ch_mock_3 | Para Pos: N/A
----------------------------------------
Книга: Mock Book A | Метод: dense
Фрагмент: Full chapter returned (refinement disabled)...
Provenance: Gutenberg ID: 1001 | Chapter ID: ch_mock_1 | Para Pos: N/A
----------------------------------------

--- Режим: HYBRID ---
Книга: Mock Book A | Метод: hybrid
Фрагмент: Full chapter returned (refinement disabled)...
Provenance: Gutenberg ID: 1001 | Chapter ID: ch_mock_1 | Para Pos: N/A
----------------------------------------
Книга: Mock Book C | Метод: hybrid
Фрагмент: Full chapter returned (refinement disabled)...
Provenance: Gutenberg ID: 1003 | Chapter ID: ch_mock_3 | Para Pos: N/A
----------------------------------------

--- Режим: REFINED ---
Книга: Mock Book A | Метод: refined
Фрагмент: Fragment not available....
Provenance: Gutenberg ID: 1001 | Chapter ID: ch_mock_1 | Para Pos: -1
----------------------------------------
Книга: Mock Book C | Метод: refined
Фрагмент: Fragment not available....
Provenance: Gutenberg ID: 1003 | Chapter ID: ch_mock_3 | Para Pos: -1
----------------------------------------