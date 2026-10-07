"""Content-based recommendation engine for Project 3 (AI Recommendation Logic).

Implements the 4-step ranking pipeline described in the Decode Labs training
material:

    Ingestion -> Scoring -> Sorting -> Filtering (Top-N)

Items (job roles) and the user profile are mapped into one shared vocabulary
of skill tags, weighted with TF-IDF, and compared with cosine similarity.
No third-party libraries are used, so every step of the math is visible.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter

DEFAULT_TOP_N = 3
MIN_PREFERENCES = 3
INPUT_ATTEMPTS = 3


def load_items(filepath: str) -> list[dict]:
    """Load the item catalog from a CSV with `role` and `skills` columns.

    Skill tags are normalized (lowercase, trimmed, de-duplicated) so the
    catalog and the user profile always share the same vocabulary format.
    The original text is kept in `skills_text` for display purposes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Item catalog not found: {filepath}")

    items: list[dict] = []
    with open(filepath, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "role" not in fieldnames or "skills" not in fieldnames:
            raise ValueError("Catalog must contain 'role' and 'skills' columns.")

        for row in reader:
            role = (row.get("role") or "").strip()
            skills_text = (row.get("skills") or "").strip()
            skills = normalize_preferences(skills_text)
            if not role or not skills:
                continue
            items.append(
                {"role": role, "skills": skills, "skills_text": skills_text}
            )

    if not items:
        raise ValueError(f"Catalog contains no usable items: {filepath}")
    return items


def normalize_preferences(raw_preferences: str | None) -> list[str]:
    """Turn raw comma-separated input into clean, unique skill tags.

    Handles extra whitespace, mixed capitalization, empty fragments and
    duplicate entries. Order of first appearance is preserved.
    """
    if not raw_preferences:
        return []

    preferences: list[str] = []
    seen: set[str] = set()
    for fragment in str(raw_preferences).split(","):
        tag = fragment.strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            preferences.append(tag)
    return preferences


def catalog_vocabulary(items: list[dict]) -> list[str]:
    """Every distinct skill tag that exists in the catalog."""
    return sorted({tag for item in items for tag in item["skills"]})


def find_unknown_skills(
    raw_preferences: str | None, vocabulary: list[str]
) -> list[str]:
    """Preferences that are not catalog tags, so they cannot be scored."""
    known = set(vocabulary)
    return [
        tag for tag in normalize_preferences(raw_preferences) if tag not in known
    ]


def vectorize(
    tags: list[str], vocabulary: list[str], idf: list[float]
) -> list[float]:
    """Build one TF-IDF weighted vector inside the shared vocabulary space.

    TF(t, d) = occurrences of t in d / total tags in d
    weight   = TF * IDF, stored at the position of t in the vocabulary.

    Tags outside the vocabulary have no dimension and are ignored.
    """
    index = {tag: position for position, tag in enumerate(vocabulary)}
    counts = Counter(tags)
    total_tags = sum(counts.values())

    vector = [0.0] * len(vocabulary)
    if total_tags == 0:
        return vector

    for tag, count in counts.items():
        position = index.get(tag)
        if position is not None:
            vector[position] = (count / total_tags) * idf[position]
    return vector


def fit_tfidf(items: list[dict]) -> tuple[list[str], list[float], list[list[float]]]:
    """Fit TF-IDF weights on the item catalog only.

    Returns (vocabulary, idf, item_vectors). Learning IDF from the catalog
    keeps scores stable: the same profile always scores the same against the
    same catalog.
    """
    document_frequency: Counter[str] = Counter()
    for item in items:
        document_frequency.update(set(item["skills"]))

    vocabulary = sorted(document_frequency)
    total_documents = len(items)
    idf = [
        math.log10(total_documents / document_frequency[tag])
        for tag in vocabulary
    ]
    item_vectors = [
        vectorize(item["skills"], vocabulary, idf) for item in items
    ]
    return vocabulary, idf, item_vectors


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Cosine similarity of two equally sized vectors, in the range 0..1.

    A zero vector (no known preferences) has no direction, so it returns 0.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def recommend(
    items: list[dict],
    raw_preferences: str | None,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """Rank catalog items against a raw comma-separated preference string.

    Returns at most `top_n` results sorted by descending match score, with
    ties broken alphabetically by role. Items with a score of 0 are dropped,
    so an empty list means "nothing matched".
    """
    preferences = normalize_preferences(raw_preferences)
    if not items or not preferences or top_n < 1:
        return []

    vocabulary, idf, item_vectors = fit_tfidf(items)
    user_vector = vectorize(preferences, vocabulary, idf)

    results: list[dict] = []
    for item, item_vector in zip(items, item_vectors):
        score = cosine_similarity(user_vector, item_vector)
        if score <= 0.0:
            continue
        known_skills = set(item["skills"])
        results.append(
            {
                "role": item["role"],
                "score": score,
                "skills_text": item["skills_text"],
                # Sorted so the report never depends on the order the
                # preferences were typed in.
                "matched_skills": sorted(
                    tag for tag in preferences if tag in known_skills
                ),
            }
        )

    results.sort(key=lambda result: (-result["score"], result["role"]))
    return results[:top_n]


def ask_preferences() -> str | None:
    """Read preferences from the terminal, requiring at least three skills.

    Returns the raw input string, or None if the user cancelled or never
    provided enough input.
    """
    print(
        f"Enter at least {MIN_PREFERENCES} skills or interests, comma-separated."
    )
    print("Example: Python, Cloud, Automation\n")

    for _ in range(INPUT_ATTEMPTS):
        try:
            raw_input_value = input("Your skills > ")
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled. Exiting.")
            return None

        if len(normalize_preferences(raw_input_value)) >= MIN_PREFERENCES:
            return raw_input_value
        print(
            f"Please enter at least {MIN_PREFERENCES} skills "
            "so the matcher has enough signal."
        )

    print("No valid preferences received. Exiting.")
    return None


def main() -> None:
    """Command-line interface: ingestion, scoring, sorting, filtering."""
    print("=== AI Recommendation System ===")
    print("Project 3 - Content-Based Tech Stack Recommender (Decode Labs)\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_skills.csv")

    try:
        items = load_items(data_path)
    except (OSError, ValueError) as error:
        print(f"Could not load the item catalog: {error}")
        return

    raw_preferences = ask_preferences()
    if raw_preferences is None:
        return

    unknown = find_unknown_skills(raw_preferences, catalog_vocabulary(items))
    if unknown:
        print("\nIgnored skills that are not in the catalog: " + ", ".join(unknown))

    results = recommend(items, raw_preferences, top_n=DEFAULT_TOP_N)

    print("\nTop Recommendations:")
    if not results:
        print("No matching roles found. Try skills that appear in data/raw_skills.csv.")
        return

    for position, result in enumerate(results, start=1):
        print(f"{position}. {result['role']}")
        print(f"   Match Score: {result['score']:.4f}")
        print(f"   Matched skills: {', '.join(result['matched_skills'])}")
        print(f"   Skills: {result['skills_text']}\n")


if __name__ == "__main__":
    main()
