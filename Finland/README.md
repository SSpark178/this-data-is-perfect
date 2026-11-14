# Finland — Pipes, Leaks, and the Graph Underneath

**Flag: `SIGMOID_HOTLINE`**

## Method

Start with the **838 route-log records** in [`routes.csv`](routes.csv). Normalize town names by stripping whitespace and trailing periods, casefolding, and removing diacritics for comparison. Unify Finnish and Swedish names for the same municipality (e.g. `Helsingfors` → `Helsinki`, `Torneå` → `Tornio`, `Närpes` → `Närpiö`) using the 61-entry alias table in [`solve.py`](solve.py).

Represent each route as an **undirected pair** of normalized towns, and put these pairs into a set. This deduplicates retranscribed routes, reversed directions, offices, transport modes, and dates. Compute each city's degree from the unique edges, sort by degree descending, and read the first letters of the top seven city names.

## Validation

- **116** unique towns (matches organizer check)
- **448** unique undirected connections (matches organizer check)
- No degree ties among the top eight towns

## Ranking

| Rank | Town | Degree | Letter |
|---|---|---:|:---:|
| 1 | Helsinki | 39 | H |
| 2 | Orivesi | 38 | O |
| 3 | Tornio | 33 | T |
| 4 | Lieksa | 29 | L |
| 5 | Imatra | 27 | I |
| 6 | Nokia | 23 | N |
| 7 | Eura | 22 | E |
| 8 | Närpiö | 21 | — |

The top-seven initials produce **HOTLINE**, hence **`SIGMOID_HOTLINE`**.

## Reproduce

```bash
pip install pandas
python solve.py
```

All data and the code are included in this folder. The task's submission requires only the flag. [Official data source](https://drive.google.com/drive/folders/1vmtpg82kdxeFPGqs9IqUp80enMTYUTOR?usp=sharing).
