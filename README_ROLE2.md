# Role 2 Deliverable — Semantic Representation and Dense Retrieval Lead

Реализация всех тасков Role 2 (2.1–2.8) из `tasks_by_role.md`, готовая к
интеграции с реальным корпусом Role 1.

## Что где лежит

| Таск | Файл | Статус |
|---|---|---|
| 2.1 Стратегия представления глав | `src/representation.py` | ✅ реализовано, 2 стратегии |
| 2.2 Embedding-модель + генерация | `src/embed_chapters.py` | ✅ код готов к продакшену; см. "Про MockEncoder" ниже |
| 2.3 Нормализация + метрика | `src/embed_chapters.py::l2_normalize`, `src/sanity_checks.py` | ✅ реализовано + sanity check |
| 2.4 FAISS Flat index | `src/faiss_search.py` | ✅ реализовано, интерфейс `search(query, top_k)` |
| 2.5 Регистрация в registry | `src/faiss_search.py::register_dense_pipeline` | ✅ пишет `indexes/index_registry.json` |
| 2.6 Отчётность | `docs/role2_report.md`, `slides/role2_slide.md` | ✅ готово |
| 2.7 Backend `/search` | `app/backend.py` | ✅ FastAPI, тонкий прокси-слой |
| 2.8 Веб-интерфейс | `app/frontend.py` | ✅ Streamlit, все элементы из раздела 3.5 |

Плюс `src/data_loader.py` — небольшой синтетический демо-датасет (5 глав,
оригинальный текст, не из реальных книг), чтобы можно было прогнать весь
пайплайн end-to-end до того, как Role 1 отдаст настоящий корпус ≥500k глав.
`ChapterRecord`-схема в `representation.py` совместима с полями из таска 1.2
(`book_id`, `chapter_id`, `title`, `author`, `chapter_title`, `paragraphs`).

## Как запустить

```bash
pip install -r requirements.txt

# 1) собрать эмбеддинги + FAISS Flat индекс на демо-данных
cd src
python3 embed_chapters.py
python3 faiss_search.py          # строит индекс, регистрирует его, гоняет demo-запросы

# 2) sanity checks (2.1 и 2.3)
python3 sanity_checks.py

# 3) backend + frontend
cd ../app
uvicorn backend:app --reload --port 8000     # терминал 1
streamlit run frontend.py                     # терминал 2
```

## Про MockEncoder (важно)

В `src/embed_chapters.py` есть флаг `USE_MOCK_ENCODER`. Для реального индекса
он установлен в `False`, потому что индекс в `indexes/faiss_flat` построен
моделью `sentence-transformers/all-MiniLM-L6-v2`. `MockEncoder` — детерминированный хэш-based
псевдо-энкодер, нужен **только** чтобы протестировать механику всего пайплайна
(representation → embedding → normalization → FAISS → search → API → UI) без
интернета. Он **не** даёт осмысленного семантического поиска — это видно по
демо-результатам (некоторые совпадения основаны на пересечении слов, а не
смысла).

Перед реальным запуском на корпусе Role 1:
1. `pip install sentence-transformers torch`
2. убедиться, что `USE_MOCK_ENCODER = False`
3. заменить `data_loader.load_sample_chapters()` на загрузку из
   `data/processed/` (реальная схема Role 1, таск 1.2)

Код `SentenceTransformer(MODEL_NAME)` уже написан и готов к использованию —
меняется только один флаг.

## Как подключаются остальные роли

- **Role 1** (`bm25_search.py`) → используется в `app/backend.py` в режиме
  `mode=bm25` — сейчас там понятный `NotImplementedError` с указанием, какой
  таск это закрывает.
- **Role 3** (`hybrid_search.py`) → зовёт `faiss_search.search()` напрямую как
  один из двух источников кандидатов для RRF (таск 3.3).
- **Role 4** (`search_engine.py`, `paragraph_refinement.py`) → собирает все
  режимы (`bm25`/`dense`/`hybrid`/`refined`) в единый движок (таск 4.4) и
  ANN-индекс поверх векторов, которые уже лежат в
  `indexes/faiss_flat/embeddings.npy` (таск 4.1).

## Ограничение (см. docs/role2_report.md)

Representation-стратегия `title_first_middle_last` всё ещё может не поймать
сцену, находящуюся не в начале/середине/конце главы, если глава сильно
длиннее трёх окон. Если Role 3 в error analysis (таск 3.6) увидит это как
частую причину ошибок — следующая итерация переходит на overlapping windows
или averaged-window-embeddings (альтернатива из раздела 4.3).
