# SIGMOID Challenges — Code and Evidence

This public repository contains reproducible Python solutions, practice checks, and supporting evidence for SIGMOID data challenges. Each country has its own folder and instructions. **The official submission requirements differ by challenge**; see the corresponding README.

| Country | Challenge | Code and evidence |
|---|---|---|
| 🇧🇪 **Belgium** | Fry-ndship — hedge residuals spell SUPERVISION | [Belgium/](Belgium/README.md) |
| 🇨🇭 **Switzerland** | The Weak Layer — decision-tree structure decoding | [Switzerland/](Switzerland/README.md) |
| 🇪🇸 **Spain** | ¡Feliz Navidad! — fair-price souvenir board | [Spain/](Spain/README.md) · [board figure](Spain/christmas_board.svg) |
| 🇮🇸 **Iceland** | What The Data? — geothermal export cleaner | [Iceland/](Iceland/README.md) |
| 🇦🇱 **Albania** | Looks like an anomaly! — streaming detector | [Albania/](Albania/README.md) |
| 🇫🇮 **Finland** | Pipes, Leaks, and the Graph Underneath — route graph | [Finland/](Finland/README.md) |
| 🇪🇪 **Estonia** | This Data Is Perfect! — fraud evaluation audit | [Estonia/](Estonia/README.md) |

## Evidence and reproducibility

Each folder includes a country-specific README with the method, evidence obtained from practice data, execution instructions, and the original organizer's data link where applicable. Generated evidence figures are committed as text-based SVG when suitable. Large practice CSVs for the newly added countries are **not included** in the repository: download them from the official links in each README to reproduce the results.

- **Switzerland:** submit `main.py` alone; practice word `DEMO` is not the final flag.
- **Spain:** submit the recovered flag **and a link to the figure and code**.
- **Iceland:** submit `main.py` alone; its output is an exact CSV.
- **Albania:** submit `detector.py` alone; the documented 84.59/100 is a practice score, not a guarantee for the hidden stream.
- **Finland:** submit its recovered word according to the challenge rules.
- **Estonia:** submit a flag plus an audit write-up. **Important:** its README documents a grouped leave-one-out estimate (76.66%) and also the literal `LeaveOneOut()` estimate (95.94%). The challenge explicitly names row-wise `LeaveOneOut()`; this discrepancy should be resolved with the organizer before treating its proposed flag as verified.

The original input data belongs to the SIGMOID challenge organizers. The repository is not an official evaluation result.
