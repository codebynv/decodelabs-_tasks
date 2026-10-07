# Project 3 — AI Recommendation Logic

## Overview
This is the personalization phase of the Decode Labs Industrial Training Kit (Batch 2026). Projects 1 and 2 classified and interpreted data; Project 3 predicts what a user actually wants. The result is a **content-based Tech Stack Recommender**: it takes a list of a user's skills, maps them into a shared vocabulary of skill tags, scores every job role in the catalog with TF-IDF weighted cosine similarity, and prints the Top 3 matches.

Everything is built with the Python standard library, so the entire recommendation algorithm — vector mapping, TF-IDF, cosine similarity, ranking — can be read and explained line by line.

## Objective
Build a simple recommendation system based on user preferences, as defined by the Project 3 specification:

- **Goal:** Create a simple recommendation system based on user preferences.
- **Key requirements:** take user input (choices or interests), match preferences using logic or similarity, display recommended items.
- **Key skills:** logic building, pattern matching, recommendation concepts.

## Decode Labs Requirements
Taken from the Project 3 training material ("AI Recommendation Logic"):

| PDF requirement | How this project satisfies it |
| --- | --- |
| Input → Process → Output (IPO) model | `ask_preferences()` → `recommend()` → formatted Top-N list |
| Content-based filtering only (no collaborative filtering) | Items are compared by their own attributes; no other users' data is used |
| One shared vocabulary for items and user profile | `catalog_vocabulary()` builds the tag vocabulary from the catalog; both sides are vectorized with it |
| TF-IDF weighting (penalise generic tags, reward specific ones) | `fit_tfidf()` + `vectorize()` implement the PDF formulas directly |
| Cosine similarity as the closeness metric | `cosine_similarity()` returns a 0..1 alignment score |
| Ingestion must accept a minimum of three inputs | The CLI re-prompts until at least 3 skills are entered (max 3 attempts) |
| 4-step pipeline: ingestion → scoring → sorting → filtering | `recommend()` scores all roles, sorts descending, truncates to Top-N (default 3) |
| Handle the user Cold Start (empty profile → zero vector) | Empty/blank preferences return no results and print a clear message instead of crashing |
| Report Top-N, not the whole catalog | Only the 3 highest scoring roles are displayed |

## Recommendation Approach
**Content-based filtering.** The engine compares the *attributes of the items* against the *attributes requested by the user*, so it works with a tiny catalog and no interaction history (the PDF explicitly chooses this over collaborative filtering for Project 3).

The pipeline:

```
USER PREFERENCES ("Python, Cloud, Automation")
        ↓  normalize (lowercase, trim, split on commas, drop duplicates)
CLEAN TAG LIST ["python", "cloud", "automation"]
        ↓  build the user vector inside the catalog's vocabulary
TF-IDF WEIGHTED VECTORS
        ↓  cosine similarity against every role
MATCH SCORE per role (0 .. 1)
        ↓  sort descending, break ties alphabetically, keep Top 3
RANKED RECOMMENDATIONS
```

Tags that do not exist in the catalog vocabulary have no dimension, so they are ignored during scoring (and reported to the user as "ignored skills"). This is the "shared vocabulary" rule from the PDF: naming mismatches break the similarity math.

## Dataset
`data/raw_skills.csv` — a small, hand-written, human-readable catalog (named `raw_skills.csv` because the Project 3 material refers to that file name).

- **Items:** 10 job roles (`Data Scientist`, `DevOps Engineer`, `Frontend Developer`, `Mobile App Developer`, ...).
- **Attributes:** one `skills` column of comma-separated tags per role, e.g. `AWS, Docker, K8s, CI/CD, Python, Cloud, Automation, Linux`.
- **Size:** 66 tag slots resolving to **40 unique skill tags** across 10 roles.

The dataset is deliberately small so every score can be traced by hand — the point of the project is the recommendation logic, not dataset volume.

## User Preferences
The CLI asks for skills/interests as one comma-separated line:

```
Enter at least 3 skills or interests, comma-separated.
Example: Python, Cloud, Automation

Your skills >
```

`normalize_preferences()` makes the input safe:

- splits on commas,
- strips surrounding whitespace,
- lowercases everything (`Python` → `python`),
- drops empty fragments (`" , , "` → `[]`),
- removes duplicates (`"python, python, sql"` → `["python", "sql"]`).

At least 3 skills are required (PDF: "your script must accept a minimum of three user inputs"); the prompt repeats up to 3 times, then exits politely.

## Similarity/Scoring Logic
All formulas come straight from the Project 3 material. `N` = number of roles in the catalog, `df(t)` = number of roles containing tag `t`.

**1. Term Frequency** — how representative a tag is inside one item:

```
TF(t, d) = occurrences of t in d / total tags in d
```

**2. Inverse Document Frequency** — penalty for a tag that is common across the catalog:

```
IDF(t) = log10( N / df(t) )
weight(t, d) = TF × IDF
```

IDF is learned from the **item catalog only**, so a given profile always receives the same score against the same catalog.

**3. Cosine similarity** — orientation of the two weighted vectors:

```
score(u, i) = (u · i) / ( ||u|| × ||i|| )      range: 0 .. 1
```

If either vector is all zeros (no known preferences), the score is `0.0`.

**Worked example** (the 3-role fixture used in the tests):

```
Alpha Role = {alpha, beta}      Beta Role = {beta, gamma}      N = 3
df(alpha) = 1  → IDF = log10(3/1) = 0.4771
df(beta)  = 2  → IDF = log10(3/2) = 0.1761

User profile: "alpha"  → user vector = [0.4771, 0, 0, 0]
Alpha Role            → item vector = [0.5×0.4771, 0.5×0.1761, 0, 0]
                      = [0.2386, 0.0880, 0, 0]

score = (0.4771 × 0.2386) / (0.4771 × √(0.2386² + 0.0880²)) = 0.9381
```

On the real catalog the IDF effect is visible: `swift` (in 1 of 10 roles) has IDF **1.000**, `docker` (2 of 10) **0.699**, `cloud` (4 of 10) **0.398**, while `python` and `sql` (5 of 10) only **0.301**. Matching a rare, specific skill therefore aligns more strongly than matching a generic one.

**The score is a similarity/match score only.** It is not a probability, not a prediction confidence, and not a claim about what a user will definitely like.

## Ranking
`recommend()` finishes the pipeline:

1. score every role against the user vector,
2. drop roles with score `0` (nothing in common),
3. sort by **score descending**,
4. break ties **alphabetically by role name** (deterministic — equal scores always produce the same order),
5. keep the **Top 3** (`DEFAULT_TOP_N = 3`, per the PDF's "Top-N list" example),
6. return fewer than 3 results when fewer roles match — never padded with junk.

Example of a real tie: `python, sql, ml` scores `Data Scientist` and `Machine Learning Engineer` at exactly `0.4272190667707981` each, so `Data Scientist` is listed first because D < M.

## Project Structure
```
project-03-ai-recommendation/
├── src/
│   └── recommender.py        # Recommendation logic + CLI
├── data/
│   └── raw_skills.csv        # Item catalog (10 job roles)
├── tests/
│   └── test_recommender.py   # 27 automated tests
├── README.md
├── requirements.txt          # No third-party dependencies
└── .gitignore
```

Function layout inside `src/recommender.py` (logic kept separate from terminal I/O):

```
load_items()            read and normalize the CSV catalog
normalize_preferences() clean raw user input
catalog_vocabulary()    the shared tag vocabulary
find_unknown_skills()   preferences with no dimension in the catalog
vectorize()             TF-IDF vector for any tag list
fit_tfidf()             learn IDF from the catalog, vectorize every role
cosine_similarity()     0..1 alignment of two vectors
recommend()             score → sort → filter (pure logic, no printing)
ask_preferences() / main()   terminal input/output only
```

## Technologies Used
- **Python 3** — standard library only (`csv`, `math`, `collections.Counter`, `unittest`).
- **No third-party packages** — TF-IDF and cosine similarity are implemented by hand so the math is auditable; `scikit-learn`'s `TfidfVectorizer` is the industrial equivalent of the same formulas.
- **`unittest`** for automated testing.

## Installation
No dependencies to install. Python 3.10+ is sufficient.

```bash
cd project-03-ai-recommendation
pip install -r requirements.txt   # intentionally empty (stdlib only)
```

## How to Run
```bash
python src/recommender.py
```

## Example Interaction
Real output for the PDF's own example input (`Python`, `Cloud`, `Automation`):

```text
=== AI Recommendation System ===
Project 3 - Content-Based Tech Stack Recommender (Decode Labs)

Enter at least 3 skills or interests, comma-separated.
Example: Python, Cloud, Automation

Your skills > Python, Cloud, Automation

Top Recommendations:
1. DevOps Engineer
   Match Score: 0.3876
   Matched skills: automation, cloud, python
   Skills: AWS, Docker, K8s, CI/CD, Python, Cloud, Automation, Linux

2. System Administrator
   Match Score: 0.3236
   Matched skills: automation, cloud
   Skills: Linux, Networking, Automation, Bash, Security, Cloud

3. Cloud Architect
   Match Score: 0.2852
   Matched skills: automation, cloud
   Skills: AWS, Cloud, Automation, K8s, Architecture, Docker, Azure, Security
```

Edge cases observed in practice:

```text
Your skills > python, sql
Please enter at least 3 skills so the matcher has enough signal.      ← below the minimum

Your skills > time travel, teleportation, wizardry
Ignored skills that are not in the catalog: time travel, teleportation, wizardry
Top Recommendations:
No matching roles found. Try skills that appear in data/raw_skills.csv.   ← unknown preferences

Your skills >                                                           ← blank input, 3 attempts
No valid preferences received. Exiting.
```

## Testing
```bash
python tests/test_recommender.py
```

**Result: 27 tests, all passing (`OK`).** The suite checks behavior, not just existence:

| # | Requirement | Test |
| --- | --- | --- |
| 1 | Dataset loads correctly | `test_real_catalog_loads`, `test_missing_file_raises`, `test_invalid_header_raises`, `test_empty_rows_are_skipped` |
| 2 | Preferences are normalized | `test_lowercase_and_whitespace`, `test_empty_and_blank_input` |
| 3 | Matching preferences give relevant results | `test_matching_preferences_find_the_right_role`, `test_real_catalog_relevant_result` |
| 4 | Scores follow the formula | `test_score_matches_hand_computed_formula`, `test_idf_values_follow_pdf_formula`, `test_idf_penalises_common_tags`, `test_cosine_similarity_basics` |
| 5 | Results sorted by score | `test_results_are_sorted_descending`, `test_scores_are_bounded` |
| 6 | Top-N limit works | `test_top_n_limits_result_count`, `test_fewer_matches_than_requested`, `test_invalid_top_n_returns_nothing` |
| 7 | Empty preferences handled | `test_empty_preferences_return_no_results`, `test_empty_catalog_returns_no_results` |
| 8 | Unknown preferences handled | `test_unknown_preferences_return_no_results`, `test_unknown_skills_are_reported`, `test_known_skills_are_not_reported` |
| 9 | Duplicates do not distort results | `test_duplicates_are_removed`, `test_duplicate_preferences_do_not_change_scores` |
| 10 | Ordering is deterministic | `test_ordering_is_deterministic`, `test_ties_break_alphabetically`, `test_vocabulary_is_sorted_and_unique` |

## What I Learned
- **Why IDF matters more than raw overlap:** a plain "count the shared tags" matcher ranks a match on `python` (in half the catalog) the same as a match on `swift` (in one role). The `log10(N/df)` penalty is what turns that into a meaningful ranking.
- **Cosine similarity is magnitude-invariant:** adding redundant preferences cannot inflate a score, which is why duplicates are removed during normalization and results stay stable.
- **Determinism is a design decision:** sorting on `(-score, role)` rather than relying on dictionary or CSV order means the same input always produces byte-identical output (verified by hashing two runs).
- **Separation of concerns:** keeping `recommend()` free of `input()`/`print()` made it possible to test scoring, sorting and edge cases without mocking the terminal.
- **The cold start problem is real:** an empty profile is a zero vector, and a zero vector has no direction — so the system must be told what to do rather than being allowed to divide by zero.

## Limitations
- **Exact tag matching only:** `Web Design` and `Frontend Development` are different dimensions (the PDF calls this out explicitly). Synonyms such as `cloud computing` vs `cloud` are not bridged; unmatched tags are simply reported as ignored.
- **User cold start:** with no skills entered, there is nothing to score. The 3-skill prompt is a forced ingestion step, not a learned model.
- **Item cold start / IDF edge:** a tag that appeared in *every* role would get `IDF = log10(1) = 0` and could not influence ranking. The current catalog has no such tag.
- **Cosine normalization can favour a broader role:** asking for `javascript, react, css, html` ranks `Full Stack Developer` (0.7924) above `Frontend Developer` (0.7030). Both contain all four skills, but Frontend's two distinctive tags (`Web Design`, `TypeScript`) were not requested by the user and lengthen its vector, widening the angle. This is inherent to cosine similarity, not a bug — but it is the kind of result that has to be explained to a stakeholder.
- **No learning from behavior:** nothing is updated after a recommendation; every run is computed from the static catalog.
- **Scores are similarity values, not probabilities** — `0.3876` does not mean "38.76% likely to like this role".
- **Small, hand-written catalog** (10 roles) chosen for transparency rather than coverage.

## Future Improvements
- Add a small alias map (`cloud computing → cloud`, `ml → machine learning`) to soften the vocabulary mismatch problem.
- Implement the PDF's cold-start bypasses: an onboarding survey or a "trending roles" fallback when a profile is empty.
- Let users rate recommendations so the catalog can be re-weighted from feedback (mentioned in the PDF conclusion).
- Smoothed IDF (`log10((1+N)/(1+df)) + 1`) so universal tags keep a small positive weight.
- Expand the catalog with a second attribute column (e.g. `category`) and show it in the output.
