# Role 1 — Data, Preprocessing, and BM25

Short instructions for reproducing the Role 1 pipeline.

## 1. Setup

```bash
git clone https://github.com/DLS-Team/Book-search.git
cd Book-search
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

If Python 3.12 is not available:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 2. Download Gutenberg books

For the reproducible teammate version, use 5,000 books:

```powershell
python src\download_gutenberg_dataset.py --max-books 5000
```

Output:

```text
data/raw/project_gutenberg_books.jsonl
```

For a quick smoke test only:

```powershell
python src\download_gutenberg_dataset.py --max-books 100
```

## 3. Preprocess books

Run:

```powershell
python src\data_preprocessing.py
```

Outputs:

```text
data/processed/processed_chapters.jsonl
outputs/dataset_stats.json
outputs/preprocessing_failures.jsonl
```

Check the statistics:

```powershell
type outputs\dataset_stats.json
```

Important field:

```text
accepted_rows
```

It should be at least 500,000 for the project requirement.

The processed searchable object contains:

```text
book_id
title
author
chapter_id / chunk_id
chapter_title
text
paragraph pointers
char_length
token_length
```

## 4. Build BM25 index

Build the BM25 sparse baseline:

```powershell
python src\bm25_search.py build --max-docs 600000
```

Outputs:

```text
indexes/bm25/bm25_index/
indexes/bm25/bm25_registry.json
indexes/bm25/metadata.pkl
```

The default `--max-docs 600000` is enough for the project scale requirement and is safer for laptops.

## 5. Run BM25 search

Example queries:

```powershell
python src\bm25_search.py search --query "fireplace winter night" --top-k 5
```

```powershell
python src\bm25_search.py search --query "murder knife blood" --top-k 5
```

```powershell
python src\bm25_search.py search --query "ship storm sea" --top-k 5
```

On Windows, this warning may appear:

```text
resource module not available on Windows
```

It is harmless if results are printed.

## 6. Generate BM25 examples

Run:

```powershell
python src\bm25_examples.py
```

Output:

```text
outputs/bm25_examples.json
```

This file contains exact-keyword examples and BM25 semantic failure cases.

## 7. Important files

```text
src/download_gutenberg_dataset.py   # downloads Gutenberg books
src/data_preprocessing.py           # cleans and chunks books
src/bm25_search.py                  # builds and queries BM25
src/bm25_examples.py                # saves BM25 example outputs
docs/role1_notes.md                 # design notes and short explanation
```

## 8. Git rules

Do not commit generated data, indexes, caches, or virtual environments.

These should be ignored:

```text
data/raw/
data/processed/
outputs/
.venv/
*.jsonl
*.pkl
*.npy
*.faiss
```

Commit only source code and docs:

```powershell
git add src docs requirements.txt .gitignore
git commit -m "Add Role 1 data and BM25 pipeline"
git push
```

Check before committing:

```powershell
git status
```

If large files were staged accidentally:

```powershell
git restore --staged data outputs
```

## 9. Short explanation

Role 1 builds the searchable corpus and BM25 lexical baseline.

The original plan expected chapter-level Gutenberg objects. The accessible chapter-level subset was too small, so we generate stable pseudo-chapter / scene chunks from Gutenberg books.

BM25 is used as the sparse lexical baseline. It works well for exact words, names, rare terms, and literal phrases. Its main limitation is lexical overlap: it may miss relevant passages that describe the same meaning with different words.
