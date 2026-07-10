"""
Demo/test fixture ONLY.

Role 1 owns the real corpus (task 1.1) and the real metadata schema
(task 1.2). Until that is delivered, Role 2 needs *something* shaped like
Role 1's output to build and unit-test representation.py, embed_chapters.py
and faiss_search.py end-to-end.

The chapters below are short, original, synthetic scene descriptions written
for this demo (not excerpts from any real book), so they can be safely used,
printed, and shared without copyright concerns. They intentionally cover the
query categories from section 7.1 (fireplace scene, guilt/betrayal, lonely
city walk, tense pre-murder conversation, a child being brave) so the demo
search results look meaningful.

When Role 1's real `data/processed/` chapters are ready, replace
`load_sample_chapters()` with a loader over that store — the ChapterRecord
shape (book_id, chapter_id, title, author, chapter_title, full_text) must stay
compatible with Role 1's schema (task 1.2).
"""

from __future__ import annotations

from typing import List

from representation import ChapterRecord

_SAMPLE_CHAPTERS_RAW = [
    dict(
        book_id="demo-001",
        chapter_id="demo-001-ch03",
        title="The Long Winter Road",
        author="A. Demo Author",
        chapter_title="Chapter 3: The Fire",
        full_text=(
            "Snow fell steadily outside as Elena pulled her chair closer to the hearth. "
            "The old house creaked in the wind, but inside, the fire crackled warmly, "
            "casting long shadows on the walls.\n\n"
            "She wrapped her hands around a cup of tea and stared into the flames, "
            "thinking of nothing in particular, simply grateful for the warmth after "
            "the long walk home through the storm.\n\n"
            "Her grandfather used to say that a fire on a winter night was the closest "
            "thing to peace a person could buy with a few logs and a match. Tonight, "
            "for the first time in weeks, she believed him."
        ),
    ),
    dict(
        book_id="demo-002",
        chapter_id="demo-002-ch07",
        title="Brothers of the Iron Coast",
        author="B. Demo Author",
        chapter_title="Chapter 7: What I Did",
        full_text=(
            "Marcus could not look his brother in the eye. The letter was still folded "
            "in his coat pocket, the one that had cost Tomas everything.\n\n"
            "\"I did it for the family,\" he said quietly, though even as he spoke the "
            "words tasted false. Guilt sat heavy in his chest, a stone he could not "
            "put down no matter how he tried to explain himself.\n\n"
            "Tomas said nothing. He simply turned and walked into the rain, and Marcus "
            "understood that some betrayals could never be undone, only carried."
        ),
    ),
    dict(
        book_id="demo-003",
        chapter_id="demo-003-ch01",
        title="Streets of Grey",
        author="C. Demo Author",
        chapter_title="Chapter 1: Nobody's City",
        full_text=(
            "The city at midnight belonged to no one. Daniel walked alone beneath "
            "flickering streetlamps, his footsteps the only sound on the wet pavement.\n\n"
            "Every window was dark. Every door was closed. He had lived here for six "
            "years and still felt like a stranger passing through a place that had "
            "already forgotten him.\n\n"
            "He kept walking, not because he had somewhere to be, but because stopping "
            "meant admitting how alone he really was."
        ),
    ),
    dict(
        book_id="demo-004",
        chapter_id="demo-004-ch12",
        title="The Quiet Investigator",
        author="D. Demo Author",
        chapter_title="Chapter 12: Before Midnight",
        full_text=(
            "The two men sat across from each other, the single lamp between them "
            "throwing more shadow than light.\n\n"
            "\"You know why I'm here,\" said the inspector, his voice low and even. "
            "The other man's hand trembled slightly around his glass, though his face "
            "stayed perfectly still.\n\n"
            "Neither of them moved. Outside, the clock in the hall struck eleven, and "
            "somewhere in that silence, both of them understood that only one of them "
            "would leave the room the same person he had been an hour before."
        ),
    ),
    dict(
        book_id="demo-005",
        chapter_id="demo-005-ch04",
        title="Small Hands, Brave Hearts",
        author="E. Demo Author",
        chapter_title="Chapter 4: The Dark Cellar",
        full_text=(
            "Mia stood at the top of the cellar stairs, her candle throwing a small, "
            "trembling circle of light into the dark below.\n\n"
            "Her heart pounded so loudly she was sure it could be heard through the "
            "floorboards. She was afraid, more afraid than she had ever been, but her "
            "little brother was down there somewhere, and someone had to go.\n\n"
            "She took a breath, gripped the candle tighter, and stepped down into the "
            "dark, one small brave step at a time."
        ),
    ),
]


def load_sample_chapters() -> List[ChapterRecord]:
    return [ChapterRecord(**row) for row in _SAMPLE_CHAPTERS_RAW]
