# SIGMOID Challenges — Code and Evidence

This public repository contains reproducible Python solutions, practice checks, and supporting evidence for SIGMOID data challenges. Each country has its own folder and instructions. **The official submission requirements differ by challenge**; see the corresponding README.

| Country | Challenge | Code and evidence |
|---|---|---|
| 🇷🇼 **Rwanda** | The Drying Floor at Karongi — loop convergence | [Rwanda/](Rwanda/README.md) |
| 🇵🇹 **Portugal** | The Douro Paradox — Simpson's paradox on vineyard distance | [Portugal/](Portugal/README.md) |
| 🇪🇬 **Egypt** | Always Valid, No Excuses — supplied parser and output | [Egypt/](Egypt/main(2).py) · [test output](Egypt/test_output.txt) |
| 🇦🇴 **Angola** | The Waterline Replay — answer | [Angola/](Angola/answer.txt) |
| 🇧🇪 **Belgium** | Fry-ndship — hedge residuals spell SUPERVISION | [Belgium/](Belgium/README.md) · [residual evidence figure](Belgium/model_evidence.svg) |
| 🇨🇭 **Switzerland** | The Weak Layer — decision-tree structure decoding | [Switzerland/](Switzerland/README.md) |
| 🇪🇸 **Spain** | ¡Feliz Navidad! — fair-price souvenir board | [Spain/](Spain/README.md) · [board figure](Spain/christmas_board.svg) |
| 🇮🇸 **Iceland** | What The Data? — geothermal export cleaner | [Iceland/](Iceland/README.md) |
| 🇦🇱 **Albania** | Looks like an anomaly! — streaming detector | [Albania/](Albania/README.md) |
| 🇫🇮 **Finland** | Pipes, Leaks, and the Graph Underneath — route graph | [Finland/](Finland/README.md) |
| 🇬🇧 **United Kingdom** | Call My BLEUff — **SIGMOID_FRINDLOW** | [United_Kingdom/](United_Kingdom/README.md) · [semantic evidence figure](United_Kingdom/semantic_scores.svg) |
| 🇬🇷 **Greece** | Just a Thing Between Siblings — **SIGMOID_ARES** | [Greece/](Greece/README.md) · [confusion and recall chart](Greece/comparison.svg) |
| 🇩🇪 **Germany** | Cutia neagră — **SIGMOID_FACTORIAL**, 100/100 accepted | [Germany/](Germany/README.md) · [solution](Germany/main.py) |
| 🇷🇴 **Romania** | Shaken, Still Standing — **SIGMOID_CONVOLUTION** | [Romania/](Romania/README.md) · [mixture-of-experts code](Romania/solution.py) |
| 🇪🇪 **Estonia** | This Data Is Perfect! — fraud evaluation audit | [Estonia/](Estonia/README.md) |

## Evidence and reproducibility

Each folder includes a country-specific README with the method, evidence obtained from practice data, execution instructions, and the original organizer's data link where applicable. Generated evidence figures are committed as text-based SVG when suitable. Some country folders link to the organizer's data rather than committing large input CSVs; **Greece includes the test dataset, true labels, and all three PyTorch model checkpoints** for direct reproducibility.

- **Germany:** submit `main.py` alone; the platform accepted 100/100 with 136/150 queries and released `SIGMOID_FACTORIAL`.
- **Switzerland:** submit `main.py` alone; practice word `DEMO` is not the final flag.
- **Spain:** submit the recovered flag **and a link to the figure and code**.
- **Iceland:** submit `main.py` alone; its output is an exact CSV.
- **Albania:** submit `detector.py` alone; the documented 84.59/100 is a practice score, not a guarantee for the hidden stream.
- **Finland:** submit its recovered word according to the challenge rules.
- **Greece:** submit **`SIGMOID_ARES`**; all three models are 95% accurate, but Ares misses only 5 of the 100 catastrophic threats.
- **Estonia:** submit a flag plus an audit write-up. **Important:** its README documents a grouped leave-one-out estimate (76.66%) and also the literal `LeaveOneOut()` estimate (95.94%). The challenge explicitly names row-wise `LeaveOneOut()`; this discrepancy should be resolved with the organizer before treating its proposed flag as verified.

The original input data belongs to the SIGMOID challenge organizers. The repository is not an official evaluation result.
