# The local search: Wikipedia on the device instead of Wikipedia's API

The model searches by writing `<search>query</search>`, and the harness answers with `Title: text` served
256 tokens at a time. Until now that text came from the Wikipedia API (its search picks the page, its
extract is the text). `ctl/localsearch/` replaces the API with a store on the device, within 5 GB of
disk and 0.5 GB of memory, so the app searches without a network. This records what the model actually
reads, how the store is built, what the search does, and what it measured on the first shard of the dump.

## What the model reads

Of the 4848 rollouts of the search GRPO (g10), 3 asked for `<more>`. The model reads the first 256 tokens
of a page and either answers or searches again; the median served text per grounded rollout is 1247
characters. So the store keeps the opening of every article (1500 characters, cut at a sentence end) and
nothing else: that is the whole of what the search ever showed the model.

## The store (`build_store.py`, `store.py`)

The `wikimedia/wikipedia` 20231101.en dump, 41 parquet files, processed one at a time and deleted after
(the box has 18 GB free). 6,396,307 articles after dropping stubs under 80 characters. Each article is
`title\ntext`; 256 articles make one zstd frame appended to `docs.bin`; `blocks.idx` holds the frame
offsets. Reading an article is one frame (a few hundred KB) decompressed, cached. `titles.txt` holds the
titles in order. The store is about 2.1 GB.

## The index (`embed.py`, `search.py`)

**Vectors.** bge-small-en-v1.5 (33M parameters, 384 dimensions) over `title. opening text`, cut to 160
tokens. Only the sign of each dimension is stored: 48 bytes an article, 307 MB for the dump, memory-mapped.
The query stays float and is scored against the signs (asymmetric); binarising the query too was tried
first and finds the right page a third as often (recall at 96 candidates 23% against 73% on shard 0). The
float vectors are kept on the box (`emb_f16.bin`, 4.9 GB) but never shipped: the ranking re-embeds its
few dozen candidates from the stored text instead, at a fraction of a second, which saves 2.4 GB of int8
vectors. Product quantization at the same 48 bytes was implemented (`pq.py`) and measured no better than
the signs (71% against 75% at 128), so it stays an option, off.

**Words.** SQLite FTS5 over each article's title and first 300 characters, contentless (the text lives in
the store), about 1.8 GB. The model's queries are keyword lists - it learned them against Wikipedia's own
lexical search - and BM25 over this much text alone finds Wikipedia's page first 48% of the time on shard 0,
against 29% for the vectors alone.

**Search.** 64 candidates from each channel, interleaved, plus 16 title-only matches; each candidate's
opening is re-embedded and scored by cosine against the query, plus the IDF-weighted share of the query's
terms found in the title (0.4) and in the opening (0.2), plus 0.1 when the query contains the title, plus
0.2 times its normalised BM25. The best page is served as `Title: text`. `fetch(kw)` has the same shape as
the Wikipedia call, and `pool_eval.py` / `online_loop.py` switch to it with `SP_LOCAL_STORE`.

## Measured on shard 0 (156,075 articles, the oldest pages)

Two test sets built from what exists: (A) 131 corpus questions whose gold answer is an article title in
the shard, scored by whether the gold string appears in the served text; (B) 192 queries the model itself
wrote during g10, paired with the page Wikipedia's search returned for them, scored by whether the local
search returns the same page.

| candidates | ranking | B top-1 | B top-3 | A top-1 | A top-3 |
|---|---|---|---|---|---|
| binary Hamming 96 + titles | cosine + title bonus | 52% | 60% | 39% | 50% |
| asymmetric 128 + titles | cosine + title bonus | 52% | 60% | 37% | 52% |
| asymmetric 128 + titles | + lexical overlap | 58% | 70% | 41% | 56% |
| asymmetric 64 + BM25 64 + titles | + lexical overlap + BM25 | 57-59% | 69-71% | 38-44% | 53-60% |

The float embedding's own recall at 128 is 78% on B, and the union of the two channels reaches 79%, so the
remaining loss is in the ranking, not the candidates.

## Measured on the whole dump: the model on the search held-out

The measurement that matters is the model itself, searching the local store instead of Wikipedia, on the
same held-out and protocol as every number in `conversational_lineage.md` (eval300 subset, 34 x 3,
temperature 0.6, 4000 tokens). g14 (bf16), 2026-09-26:

| search | correct | grounded | searches per rollout | shards |
|---|---|---|---|---|
| Wikipedia API | 48.0% | 71% | 3.5 | 58.8 / 38.2 / 47.1 |
| local store (this) | **48.0%** | 67% | 3.7 | 47.1 / 52.9 / 44.1 |

Paired on the 51 distinct questions: both right 13, Wikipedia only 9, local only 10, neither 19. The
local search is worth the same as the API to this model, on this set, and it needs no network.

## What is not yet shown

The memory figure below is an estimate for a native implementation. The Python prototype (onnxruntime,
numpy, SQLite) sits at 1.0 GB resident on shard 0 after a few queries, most of it the runtime and the
re-embedding of ~130 candidates; the index itself is 7 MB there and 307 MB on the whole dump. The 0.5 GB
budget depends on the Swift port: the int8 model, the index memory-mapped, and re-embedding in batches.
Query latency in the prototype is 2-4 s on four CPU cores, dominated by the re-embedding; on the box's
GPU it is well under a second.

## Budget

| | disk | memory at query time |
|---|---|---|
| store (docs.bin, blocks.idx, titles.txt) | 2.1 GB | one frame cache, ~10 MB |
| sign index (emb.bin) | 0.3 GB | mapped, 0.3 GB when fully touched |
| lexical index (lex.sqlite) | 1.8 GB | SQLite page cache, tens of MB |
| bge-small (ONNX fp32 / int8) | 0.13 / 0.03 GB | 0.13 / 0.05 GB |
| total | ~4.3 GB | ~0.45 GB |

A query costs one pass over the sign index (0.7 s in numpy on 4 cores, milliseconds with SIMD or a GPU),
one FTS5 query (~80 ms), and the re-embedding of ~130 candidates (the largest cost on a CPU; on the phone
the int8 model at ~3 ms a candidate).

## The device layout, measured on the whole dump (2026-09-28)

The first version of the search read the whole sign index per query, OR-ed every query term through
FTS5, and re-embedded 128 candidates in one batch: 7 s and 1.7 GB resident a query on the whole dump.
The layout that ships:

- **IVF over the sign index** (`ivf.py`): 2048 centroids from k-means on the signs, the index rewritten in
  cluster order; a query scores the centroids (3 MB) and reads its 48 best clusters, scored 16k articles at
  a time with a running top-k. ~2% of the index is touched per query.
- **Lexical queries shaped to match few rows**: FTS5 scores every matching row, so the two rarest query
  terms are AND-ed first (1 ms), then the four rarest OR-ed (100 ms; recall@64 62% against 65% for all
  terms at 550 ms). 4 MB of SQLite page cache.
- **The int8 embedder** (33 MB against 133 MB fp32) and candidates re-embedded 16 at a time, 64 of them,
  96 tokens each.

On 200 of the model's own queries against the page Wikipedia's search returned (the whole dump, 6.4M
articles), the channels' recall at 64 candidates: embedding IVF 43% (96 probes), lexical AND 51%, lexical
OR-4 66%, title 48%, union 70%. Ranking then decides: the fused score (cosine + 0.2 title overlap + 0.4
opening overlap + 0.2 when the query contains the title; weights from a grid on these queries) puts the
page first 49% of the time, in the top three 52%. A cross-encoder rerank (MiniLM-L6, int8, 23 MB) over
the top 32 or 64 did not improve on it (43 / 53), so the search stays a bi-encoder. The remaining gap
between coverage (68-70%) and top-three (52%) is pages that share a name or a subject with the right one
- Wikipedia's own choice among them is not recoverable from a title and 400 characters.

| | per query (4 CPU cores, numpy) | resident |
|---|---|---|
| first version | 7 s | 1.7 GB |
| device layout | 0.6 s | 0.55 GB (Python; the index reads are memory-mapped and evictable) |

The end-to-end measure - the 4-bit model on the search held-out with this search in place of the API - is
what decides whether the lower coverage costs answers; the float model with the first version scored the
same as the API (48.0 / 48.0 on 102 rollouts).

## End to end with the 4-bit model (2026-09-28)

The GPTQ 4-bit model (`release/g14-4bit-gptq`) on the search held-out, 102 rollouts, its own pooler:

| search | correct | grounded | searches per rollout | shards |
|---|---|---|---|---|
| Wikipedia API | 48.0% | 72% | 3.7 | 47.1 / 47.1 / 50.0 |
| local, device layout (IVF, int8 embedder, lean ranking) | **48.0%** | 71% | 2.2 | 50.0 / 47.1 / 47.1 |

Same answers, fewer searches, and 20 s a rollout against 35-50 s with the API. At 300 rollouts, paired on
the same 150 questions: local 42.3% (43.0 / 39.0 / 45.0) against the API's 45.0% (50.0 / 46.0 / 39.0),
-2.7 +- 4.1 (better on 36 questions, worse on 45, equal on 69); grounded 67% against 63%, searches 3.2
against 4.6. Within the noise, with a hint of two or three points: the model reaches a page with the
answer more often through the local search and answers right slightly less often. Serving the two best pages
(700 characters each) in the first block read 48.0% at 102 rollouts (50.0 / 58.8 / 35.3), the same as one page at 102
and no signal either way; it was not extended. The retriever below is the next attempt.

## Where the answers are lost (2026-09-28, the 300-rollout pairing read again)

The same 300 rollouts, split by where the question's answer string sat in what the model was served:

| | in a title | in the first 300 chars | later in the block | nowhere |
|---|---|---|---|---|
| Wikipedia API | 35 rollouts, 89% right | 120, 71% | 28, 57% | 117, 3% |
| local store | 12, 83% | 144, 65% | 43, 44% | 101, 4% |

The local search reaches a text carrying the answer more often (199 rollouts against 183), but Wikipedia's
search three times as often lands on the page *titled* with the answer, and the answer it serves sits
earlier. Wikipedia searches the whole article, so the answer entity's own page (which mentions the
question's subject somewhere in its body) is reachable; the store indexes only each page's opening, so the
local search returns the subject's page and the answer is a mention inside it, which the model turns into a
right answer less often (65% against 71% when it is in the first 300 characters, 44% against 57% later).
That, not coverage, is the 2.7 points.

## A retriever of our own (2026-09-28)

bge-small fine-tuned (`train_retriever.py`) on every search event of every online run on the hub: 10,630
distinct (query, page served) pairs over 1,113 questions, labelled by whether the rollouts that read the
page were right or the served text carried the answer. 5,250 positive pairs (1,840 pages, 1,071
questions); negatives are the pages the model reached for the same question that did not help (3,863
pairs) and what the current search ranks near the right page (`train_negs.json`). The 400 test queries
and their 42 questions are out of training; the held-out is untouched. Two epochs, 330 steps, 108 s on
the 3060, in-batch accuracy 0.84.

`test_retriever.py` scores two things on the 400 test queries: the same page as Wikipedia, and whether the
page served carries the answer (the API's page carries it 53.4% of the time on these).

| ranking model (the index unchanged) | same page top-1 / top-3 | carries the answer top-1 / top-3 |
|---|---|---|
| bge-small as shipped | 40.2 / 47.5 | 62.2 / 72.2 |
| fine-tuned, ranking only | 39.0 / 47.5 | 60.2 / 72.2 |

| fine-tuned, the dump re-embedded with it (`wiki_store_ft1`, IVF rebuilt) | 37.8 / 47.0 | 60.9 / 71.7 |

Nothing moves. The trained model reaches 81% in-batch accuracy on its own pairs but the test queries
(other questions) read the same through it, as ranking model or as index: the candidates are mostly the
lexical channel's, and what the ranking needs (which of the answer-bearing pages to serve) is not in a
384-dimensional similarity. The dedicated retriever is kept on the hub (`localsearch/bge-small-ft1`,
`localsearch/wiki_en_20231101_ft1`) and the end-to-end held-out with it (gq14R) runs for the record.

**The ranking re-fitted for answer-bearing pages** (1,200 training queries with the question's answer
known, features per candidate: cosine of either model, title overlap, opening overlap, query-names-title,
BM25; grid and a listwise linear fit): 62.2% → 63-65% top-1 on the 400 test queries, inside the noise of
399 queries. The candidates hold an answer-bearing page 81% of the time; no linear ranking over these
features reaches it more often than about 64%. Pulling the query's sentence forward in the served text was
also measured and dropped: in only 3 of the 300 local rollouts did the page carry the answer beyond the
block the model read.
