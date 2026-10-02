# Sentence-sequence BART (page vectors from sentence vectors)

A transformer over the sequence of a document's sentence vectors (bge-small-en-v1.5, CLS, 384-d, int8 on disk):
an encoder reads the document with sentences hidden, a causal decoder regenerates it one sentence vector at a time,
and a page token in front of the encoder gives one vector per document. Code: `ctl/sentbart/` (prep.py, embed.py,
train.py); box control: `ctl/boxI.sh`. Data: English Wikipedia 20231101, shards 0-15 (16 of 41, ~1.6M articles);
shard 001 held out (118,398 articles).

## Training objective (unchanged since run4)

- encoder: hidden sentences (30% in spans, or the tail) -> their vectors (InfoNCE against the batch's sentences + cosine)
- decoder: next sentence vector from the ones before it (same loss)
- page: one sentence hidden per document; its raw bge vector must find its own document's page vector (InfoNCE,
  in-batch pages + the last 64 batches' pages as negatives, `--page-queue 64`)

## Results (held-out shard; page retrieval: 2000 documents, one sentence hidden in each, that sentence finds its page)

| run | what | page top-1 | page top-10 | 90% within | next-sentence top-10 |
|---|---|---|---|---|---|
| mean of the other sentences' vectors (no model) | baseline | 56.9 | 78.8 | top 45 | - |
| run4 | page token, in-batch negatives, 200k | 39.0 | 76.5 | top 35 | 73.1 |
| run5 | page = mean + learned correction, 40k | 58.3 | 81.7 | top 36 | 67.4 |
| run4c | run4 + 64 batches of negatives, 100k | 49.3 | 78.8 | top 34 | - |
| run4c2 | run4c continued at lr 3e-4 | diverged at ~40k (gradient 1 -> 1e10), discarded | | | |
| **run4d1** | run4c continued at lr 1e-4, skip guard, best kept | **51.1** | 79.7 | - | - |
| run4d2 | run4d1 continued | 51.1 at 40k (no gain) | | | |
| run4g* | run4d1 grown 4+4 -> 8+8 layers (identity-initialised new layers) | running | | | |

Taken without training on run4c: the mean of the encoder's outputs through page_head 43.5, the mean of the encoder's
own sentence reconstructions 34.5 - the page token is the right read-out.

With the app model's own `<search>` queries (bge query instruction) over all 118,398 held-out articles, a single
vector per article: page vector top-100 40.5% (pages Wikipedia served) / 47.5% (pages that held the answer); the lead
sentence 70.5 / 86.9; the mean 65.1 / 78.7; the best single sentence (not storable on a device) 78.8 / 100. The page
vector is not trained for keyword queries; search would put an MLP / end-to-end stage on top, or use the lead vector.

## Files on the hub (`baya1116/hypernet-sp-distill`, `sentbart/`)

- `small_run4d1/model_best.pt` - the best 4+4 model (d 512, 8 heads, FFN 2048, seq 128), `{"model": state_dict, "args", "step"}`;
  load with `ctl/sentbart/train.py` (`--init`, `--eval-only 1` measures it). Also `small_grow_source.pt` (the same weights).
- `small_run*/train.log` - every evaluation; `searcheval/` - the article-search test's queries and results.
