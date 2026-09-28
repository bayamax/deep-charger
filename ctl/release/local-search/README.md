# Release: the local search, 2026-09-28

Wikipedia on the device in place of Wikipedia's API: the model's `<search>query</search>` is answered from
this directory, without a network. English Wikipedia, dump 20231101, 6,396,307 articles (stubs under 80
characters dropped), the opening 1,500 characters of each (what the model ever reads: it takes the first
256 tokens of a page and searches again or answers).

| file | what | size |
|---|---|---|
| `docs.bin`, `blocks.idx`, `titles.txt`, `meta.json` | the store: `title\ntext` records, 256 to a zstd frame, frame offsets, titles in order | 2.3 GB |
| `emb.bin` | bge-small-en-v1.5 vectors of `title. opening`, sign bits only (48 bytes an article) | 0.3 GB |
| `ivf_centroids.npy`, `ivf_order.npy`, `ivf_offsets.npy`, `emb_ivf.bin` | the sign index in cluster order (2048 centroids): a query reads its 48 nearest clusters, ~2% of the index | 0.3 GB |
| `lex.sqlite` | FTS5 over title + first 300 characters, contentless (the text is in the store) | 1.5 GB |
| `bge-small-en-v1.5/` | the embedder: `model_int8.onnx` (33 MB), tokenizer | 0.04 GB |
| `code/` | `store.py`, `search.py`, `ivf.py` and the build scripts | |

Total on disk about 4.4 GB (with the IVF copy of the sign index; `emb.bin` can be dropped when
`emb_ivf.bin` is present). Memory: 0.29-0.37 GB resident as its own process on the whole dump (numpy,
onnxruntime, SQLite; the index files are memory-mapped and evictable). A query is 0.6 s on four CPU
cores (1.2-1.4 s on the box's CPU beside the model): the query embedded, the IVF probe, two FTS5 queries
(the two rarest terms AND-ed, then the four rarest OR-ed), 64 candidates re-embedded from their stored
openings (96 tokens each, 16 at a time) and ranked by cosine + 0.2 title overlap + 0.4 opening overlap +
0.2 when the query names the title. `LocalSearch(store, embedder).fetch(query)` returns the same
`Title: text` the API path served.

Measured with the released 4-bit model (`release/g14-4bit-gptq`) on the search held-out (eval300 subset,
temperature 0.6, 4000 tokens, same protocol as the model's numbers):

| search | 102 rollouts | 300 rollouts |
|---|---|---|
| Wikipedia API | 48.0% | 45.0% |
| this directory | 48.0% | 42.3% |

Paired over the same 150 questions at 300: -2.7 +- 4.1 (better on 36, worse on 45, equal on 69). The
model reaches a text carrying the answer more often through this search (199 of 300 rollouts against
183 with the API) but Wikipedia's search, which reads whole articles, lands three times as often on the
page *titled* with the answer, which the model turns into a right answer more reliably. What was tried
and did not move it: two pages in the first block, a retriever fine-tuned on the model's own queries
(`localsearch/bge-small-ft1`, `localsearch/wiki_en_20231101_ft1`: the same numbers), the ranking
re-fitted for answer-bearing pages. The repository's `docs/local_search.md` has the measurements.
