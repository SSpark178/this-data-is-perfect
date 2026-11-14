# Mongolia — O sarcină bee-ton!

Candidatul rezultat din decodare este **`SIGMOID_RECALL`**. Confirmarea oficială aparține evaluatorului concursului.

Pentru trimitere, folosiți URL-ul acestui folder și flagul de mai sus. `counts.csv` conține câte o estimare întreagă, nenegativă pentru fiecare dintre cele 333 de fotografii de survol.

## Fișiere

- [`counts.csv`](counts.csv): rezultatul cerut, cu antetul `image,count`.
- [`counts_details.csv`](counts_details.csv): estimările înainte de rotunjire, metoda folosită și informații despre fotografia donatoare, când există o potrivire geometrică.
- [`main.py`](main.py), [`density.py`](density.py), [`matching.py`](matching.py), [`decode.py`](decode.py): codul soluției.
- [`validation.csv`](validation.csv): predicții și numărători reale pentru 56 de imagini de validare.
- [`survey_map.svg`](survey_map.svg): harta numărătorilor estimate, cu valorile afișate în fiecare celulă.
- [`decoder_candidates.csv`](decoder_candidates.csv): candidații OCR ordonați după scor.
- [`decoded_word.svg`](decoded_word.svg): reprezentarea în font 5×7 a candidatului OCR ales; aceasta este o reconstrucție OCR, separată de harta măsurătorilor.

## Date

Datele sunt cele furnizate de organizatori:

https://drive.google.com/drive/folders/1acU-wiKH9ar7uTmg3m8-v2aAtEvT4k9e

Fotografiile și adnotările se descarcă folosind opțiunea `--download`. Nu sunt incluse în repository.

## Metoda de numărare

1. Redimensionez imaginile la 150×150 pentru estimarea densității. Din fiecare imagine extrag culori BGR/Lab/HSV, medii și variații locale, gradient și laplacian, la cinci scări.
2. Transform fiecare centru de oaie din `train_points.csv` într-o contribuție Gaussiană cu masă unitară. Un estimator Extra Trees cu 48 de arbori învață densitatea locală din 1.200 de pixeli eșantionați per fotografie.
3. Calibrez numărătoarea la nivel de imagine folosind statisticile caracteristicilor, distribuția densității și componentele conexe la mai multe praguri. Calibratorul este un estimator Extra Trees cu 300 de arbori.
4. Caut și fotografii care se suprapun cu fotografiile adnotate. Folosesc RootSIFT, un vocabular vizual pentru recuperarea candidaților, căutare FLANN și omografii RANSAC, inclusiv pentru fotografii oglindite.
5. Accept transferul punctelor când există cel puțin 40 de inlieri, proporție de inlieri ≥0,35, acoperire ≥0,30 și corelație fotometrică ≥0,40. Număr punctele adnotate proiectate în fotografie și adaug densitatea estimată din regiunea neacoperită. Pentru celelalte imagini folosesc numărătoarea calibrată.

Numărătorile sunt estimări și pot conține erori, în special în turme dense sau la marginile fotografiilor.

## Validare

Împărțirea folosește `numpy.random.default_rng(178)`: 56 de imagini pentru validare și 224 pentru antrenarea modelelor de validare. Fotografiile donatoare pentru validare provin din cele 224 de imagini de antrenare.

Rezultatele înregistrate în dezvoltare:

| Metodă | Eroare absolută medie, oi/imagine |
|---|---:|
| Suma densității locale | 11,30 |
| Numărătoare calibrată | 5,47 |
| Transfer geometric, cu estimarea calibrată ca rezervă | 3,29 |

Ultima coloană din `validation.csv` permite recalcularea erorii de 3,29. Această împărțire a fost folosită și la alegerea parametrilor; reprezintă o măsurare de validare, nu o evaluare independentă a survolului.

Pentru predicția finală, baza de densitate a calibratorului rămâne antrenată pe cele 224 de imagini, iar calibratorul folosește numărătorile tuturor celor 280 de imagini. Transferul folosește punctele tuturor celor 280 de fotografii. Pentru regiunile neacoperite folosesc și estimatorul de densitate antrenat pe toate cele 280 de imagini.

## Decodare

PCA aplicat coordonatelor din `survey_plots.csv` identifică axele unei grile rotite de 37×9, cu pas de aproximativ 50 m. Așez numărătorile în această grilă.

Grila se potrivește unei inscripții de șase caractere în font 5×7, cu margine și spațiere de o celulă. Din cauza erorilor de numărare, potrivirea fiecărui caracter separat produce o imagine zgomotoasă. Decoderul compară șabloanele cu valorile grilei și folosește dicționarul general CMU ca prior lingvistic. Verifică și orientările oglindite ale grilei. Pragul este ales prin Otsu, iar contribuția valorilor foarte mari este limitată pentru robustețe.

Candidatul cu scor maxim este **RECALL**. Același cuvânt a fost candidatul principal în două variante ale numărătorii bazate pe transfer geometric. Harta brută și lista candidaților sunt păstrate pentru verificare.

## Reproducere

Din folderul `Mongolia`, cu Python 3.12 sau o versiune compatibilă:

```bash
pip install -r requirements.txt
python main.py --download
```

Dacă datele sunt deja descărcate:

```bash
python main.py --data /cale/catre/date
```

Structura datelor trebuie să fie:

```text
data/
  train/
  survey/
  train_counts.csv
  train_points.csv
  survey_plots.csv
```

Prima rulare antrenează modelele și construiește indexul fotografic. Cache-ul este reutilizat la rulările următoare. Algoritmii aproximativi FLANN/RANSAC pot produce mici variații ale numărătorilor între versiuni de biblioteci și rulări.

Pentru a repeta doar decodarea fișierului publicat:

```bash
python decode.py --counts counts.csv --plots data/survey_plots.csv
```
