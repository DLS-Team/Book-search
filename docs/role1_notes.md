# Role 1 — Data, Tokenization, and BM25 Baseline

## Corpus construction

The original plan used chapter-level Gutenberg objects. During implementation, the accessible chapter-level Hugging Face subset contained too few objects for the required retrieval scale. Therefore, the project uses a fallback pipeline over Project Gutenberg books and splits them into stable pseudo-chapter / scene chunks.

The searchable object is a stable text chunk with:

- `book_id`
- `title`
- `author`
- `chapter_id` / `chunk_id`
- `chapter_title`
- `text`
- paragraph-position pointers
- character length
- token length

This keeps every result traceable to its source while providing enough searchable objects for large-scale BM25, dense, ANN, and hybrid retrieval experiments.

## Dataset statistics

The latest processed corpus was generated from **5,000 raw books**.

| Statistic | Value |
|---|---:|
| Accepted searchable objects | 553,472 |
| Rejected items | 2,457 |
| Removed percentage | 0.442% |
| Books after processing | 3,150 |
| Authors after processing | 2,254 |
| Average characters per object | 2,165.85 |
| Median characters per object | 1,961 |
| Average tokens per object | 385.74 |
| Median tokens per object | 348 |
| Average paragraph pointers per object | 5.75 |
| Median paragraph pointers per object | 5 |

The final searchable corpus contains **553,472 stable pseudo-chapter / scene chunks**.

## Chunking policy

The current preprocessing configuration is:

| Parameter | Value |
|---|---:|
| Target words per chunk | 300 |
| Maximum words per chunk | 450 |
| Minimum characters | 300 |
| Minimum tokens | 50 |
| Boundary policy | chunks end only at paragraph boundaries |

Chunks end only at paragraph boundaries. This avoids cutting text in the middle of a paragraph and improves readability in the final search interface.

Each chunk also stores paragraph-position provenance so that retrieved results can be traced back to their original position in the source book.

## Text cleaning

The preprocessing pipeline performs the following cleaning steps:

- removes Project Gutenberg boilerplate;
- converts line breaks in stored text to spaces;
- joins words split by hyphenated line breaks;
- removes separator lines;
- removes transcriber notes;
- removes illustration notes;
- removes repeated formatting symbols.

These steps reduce distracting artifacts that could negatively affect BM25 tokenization, dense embeddings, and the readability of returned fragments.

## Tokenization decision

For BM25, text is lowercased and tokenized with a simple regular-expression tokenizer.

The current policy is:

- lowercase: `true`
- punctuation: removed except apostrophes inside words
- stemming: `false`

Stemming is not used in the first iteration. This keeps the lexical baseline simple and helps preserve names, locations, and literary phrases.

## BM25 baseline

BM25 is used as the sparse lexical baseline. It is important because dense retrieval can blur exact names, rare terms, and literal phrases, while BM25 performs well when the query and passage share important vocabulary.

The BM25 index is built over the processed searchable chunks and serves two purposes:

1. a strong classical retrieval baseline;
2. the lexical branch of the hybrid BM25 + dense RRF pipeline.

## Provenance

Every result preserves identifiers and position metadata, including:

- Gutenberg book ID;
- stable chunk/chapter ID;
- title and author;
- chapter title;
- paragraph-position pointers.

This allows the serving layer to resolve a retrieved ID into readable text and show the source of every result.

## Limitations

BM25 depends on lexical overlap. It performs well for exact names, rare words, and literal descriptions, but it can miss semantically relevant scenes expressed with different wording.

Chunking also introduces a trade-off between retrieval granularity and context. Although the current boundary policy avoids splitting paragraphs, a chunk may still omit relevant context that appears in neighboring chunks.

These limitations motivate the dense, ANN, and hybrid retrieval components implemented by the other project roles.
