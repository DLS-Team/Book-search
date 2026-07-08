# Role 1 — Data, Tokenization, and BM25 Baseline

## Corpus construction

The original plan used chapter-level Gutenberg objects. During implementation, the accessible chapter-level Hugging Face subset contained only 305 chapter rows, which was far below the required scale. Therefore, we used a fallback pipeline over Project Gutenberg books and split them into stable pseudo-chapter / scene chunks.

The resulting searchable object is a stable text chunk with:
- book_id
- title
- author
- chapter_id / chunk_id
- chapter_title
- text
- paragraph position pointers
- character length
- token length

This preserves the required provenance format while satisfying the scale requirement.

## Dataset statistics

Final processed corpus:

- Raw books used: 20,000
- Accepted searchable objects: 3,662,966
- Books after processing: 12,483
- Authors after processing: 7,104
- Average tokens per object: 384.5
- Median tokens per object: 348
- Removed items: 4,504
- Removed percentage: 0.1228%

## Tokenization decision

For BM25, text is lowercased and tokenized with a simple regex tokenizer. Punctuation is removed except apostrophes inside words. Stemming is not used in the first iteration.

Reason: the first BM25 version should be a stable lexical baseline. Avoiding stemming helps preserve names, places, and literary phrases.

## BM25 baseline

BM25 is used as the sparse lexical baseline. It is important because dense search can blur exact names, rare words, and literal phrases. BM25 gives the team a strong classical baseline and a future input for hybrid retrieval.

The BM25 index was built over 600,000 processed objects for the first scalable experiment. This satisfies the project scale requirement while staying manageable on a laptop.

## Limitation

The main limitation is that BM25 depends on lexical overlap. It works well when the query and passage share exact words, but it can miss semantically relevant scenes that use different wording.