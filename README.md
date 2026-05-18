# Master Thesis Code Repository

This repository contains the analysis pipeline for my Master's thesis on science-related conspiracy narratives on Twitter.

The thesis examines how science-related conspiracy narratives vary in terms of visibility, framing, and sentiment over time. The code operationalizes a Standing-Framing-Positioning (SFP) framework using dictionary-based narrative annotation, monthly standing analysis, BERTopic-based framing analysis, VADER sentiment analysis, and integrated temporal analysis.

## Purpose of This Repository

This repository is intended to make the computational workflow transparent and reusable. It documents how the analysis was conducted, how intermediate outputs were generated, and how the main empirical figures and tables can be reproduced from the required input data.

The original raw dataset is not included in this repository due to data access and redistribution restrictions.

## Repository Structure

```text
data/
  raw/                         # local raw data, not tracked by git
  clean/                       # generated cleaned data, not tracked by git
  processed/                   # generated processed data, not tracked by git
  dictionaries/                # dictionary files used for narrative annotation

scripts/
  01_clean_data.py
  02_dictionary_smoke_test.py
  03_dictionary_label_full.py
  04_absolute_standing.py
  05_relative_standing.py
  06_dominance_competition.py
  07_bertopic_framing.py
  08_frame_architecture_types.py
  09_frame_evolution.py
  10_sentiment_baseline.py
  11_sentiment_standing_alignment.py
  12_integrated_sfp_dynamics.py

results/                       # generated analysis outputs, not tracked by default
figures/                       # generated figures
models/                        # generated BERTopic models, not tracked by default

```
## Analysis Pipeline
Run the scripts in numerical order.
```
python scripts/01_clean_data.py
python scripts/02_dictionary_smoke_test.py
python scripts/03_dictionary_label_full.py
python scripts/04_absolute_standing.py
python scripts/05_relative_standing.py
python scripts/06_dominance_competition.py
python scripts/07_bertopic_framing.py
python scripts/08_frame_architecture_types.py
python scripts/09_frame_evolution.py
python scripts/10_sentiment_baseline.py
python scripts/11_sentiment_standing_alignment.py
python scripts/12_integrated_sfp_dynamics.py
```
## Thesis Section Mapping
```angular2html
01_clean_data.py                    -> Section 3.3
02_dictionary_smoke_test.py          -> Section 3.4
03_dictionary_label_full.py          -> Sections 4.1.1 and 4.1.2
04_absolute_standing.py              -> Section 4.2.1
05_relative_standing.py              -> Section 4.2.2
06_dominance_competition.py          -> Section 4.2.3
07_bertopic_framing.py               -> Section 4.3.1
08_frame_architecture_types.py       -> Section 4.3.2
09_frame_evolution.py                -> Section 4.3.3
10_sentiment_baseline.py             -> Section 4.4.1
11_sentiment_standing_alignment.py   -> Section 4.4.2
12_integrated_sfp_dynamics.py        -> Section 4.4.3
```

## Data Requirements
The pipeline expects the raw dataset to be placed locally at:
```angular2html
data/raw/conspiracy_tweets.csv
```

This file is not included in the repository.

The dictionary file should be placed at:
```angular2html
data/dictionaries/science_related.yaml
```

## Reproducibility Note
This repository provides the computational workflow used for the thesis. It is designed to reproduce the analytical process rather than guarantee byte-for-byte reproduction of every figure in the thesis.

Some outputs, especially those based on BERTopic, UMAP, and HDBSCAN, may vary slightly across machines, dependency versions, and runs. Although fixed random seeds are used where possible, topic modeling and clustering can remain sensitive to numerical and environment differences. As a result, frame labels, topic IDs, dominant-frame transitions, and downstream integrated SFP figures may not be identical to the figures reported in the thesis.

The thesis text and its reported figures should be treated as the authoritative results from the original analysis run. This repository is provided to make the workflow transparent, inspectable, and reusable.

## Environment
Recommended Python version:
```angular2html
Python >= 3.10
```

## Install dependencies:
```angular2html
pip install -r requirements.txt
```


## Outputs
The scripts generate cleaned data, processed data, analysis tables, figures, and BERTopic model files. Most generated outputs are excluded from version control to avoid uploading large files, raw text, or restricted data-derived artifacts.

Selected aggregate tables or representative figures may be included if they do not contain raw tweet text or restricted data.