# "Lived Space": A Computational Study of Setting in Fiction

Code and data for

> Katrin Rohrbacher, "'Lived Space': A Computational Study of Setting in Fiction." In Robin M. Aust, Giulia Grisot, and Berenike Herrmann (eds.), *Comparing Landscapes. Approaches to Space and Affect in Literary Fiction*. Bielefeld University Press (forthcoming).

The paper uses a BERT classifier, fine-tuned on hand-annotated sentences, to sort sentences in 4,784 German-language works of fiction (1780–1940) into Gerhard Hoffmann's types of setting, plus a fourth "descriptive space" category.

| Label in code / data | Type of setting |
|---|---|
| `perceived_space` | *gestimmter Raum* (space as it reflects mood and atmosphere) |
| `action_space` | *Aktionsraum* (action space) |
| `visual_space` | *Anschauungsraum* (field of vision) |
| `descriptive_space` | descriptive space |
| `no_space` | no setting |

## Contents

```
data/
  space_annotations.csv       2,879 hand-annotated sentences used to train and test the classifier
  space_counts_per_book.csv   per-book metadata and the number of sentences predicted for each type
notebooks/
  figures.ipynb               reproduces the figures in the paper (Fig. 4, Fig. 5, corpus overview)
figures/                      output of figures.ipynb (PNG and PDF)
model/
  train_classifier.ipynb      fine-tunes lkonle/fiction-gbert-large on data/space_annotations.csv
  predict.ipynb               classifies every sentence of the corpus and rebuilds space_counts_per_book.csv
```

### `data/space_annotations.csv`

| Column | Description |
|---|---|
| `filename` | source text the sentence was sampled from |
| `sentence_id` | index of the sentence in that text |
| `text` | the sentence (German) |
| `space_type` | label (see table above) |

The sentences were drawn at random from about 2,300 books of German-language fiction and annotated by the author.

### `data/space_counts_per_book.csv`

One row per book: `author_last`, `author_first`, `title`, `year`, `genre` (Projekt Gutenberg-DE genre), `filename`, the number of sentences classified as each type of setting, and `num_sents` (all sentences in the book).

## Corpus

The texts and full metadata are published separately:

> Rohrbacher, Katrin (2025). *Corp: A Corpus of German-language Fiction and Non-Fiction (1780–1930)*. Journal of Open Humanities Data. https://doi.org/10.5281/zenodo.15714828

Copyright law limits the published corpus to works up to 1930. For this study, about 320 works from the American Project Gutenberg were added, which extends the corpus to 1940. Because of copyright, `space_counts_per_book.csv` includes these works as counts only, without their full text.

## Model

The classifier was fine-tuned from [`lkonle/fiction-gbert-large`](https://huggingface.co/lkonle/fiction-gbert-large) on a 70/30 train/test split of `space_annotations.csv`. The model used in the paper is available on Hugging Face as [`katrohrb/de-setting-classifier`](https://huggingface.co/katrohrb/de-setting-classifier):

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="katrohrb/de-setting-classifier")
pipe("Er trat ans Fenster und sah hinaus auf den verschneiten Hof.")
```

## Reproducing the results

The figures can be redrawn from `data/space_counts_per_book.csv` alone:

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook figures.ipynb
```

To redo the whole pipeline, which needs a GPU and `transformers>=4.41` and `torch`:

1. `model/train_classifier.ipynb` trains the classifier and prints the per-class scores (Table 2). The train/test split used for the published model was not seeded, so a new run gives slightly different numbers.
2. `model/predict.ipynb` reads the corpus text files from Zenodo (one sentence per line), classifies each sentence, and writes per-book counts in the format of `data/space_counts_per_book.csv`. Genre labels come from that file, because the published fiction metadata does not include them.

The prediction notebook truncates sentences longer than 512 tokens. The original run skipped the handful of sentences that were too long, so counts for a few books can differ by one or two sentences.
