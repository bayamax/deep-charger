"""Chat with Qwen3-1.7B-4bit-DWQ, optionally answering from your own documents (retrieval-augmented generation).

  pip install mlx-lm                      # Mac (Apple silicon): real 4-bit, about 1.4 GB
  pip install torch transformers          # elsewhere (the 4-bit weights are expanded to 16-bit: about 3.4 GB)

  python rag_chat.py                                   # plain chat
  python rag_chat.py --docs ./my_docs                  # answer from the .txt / .md files in ./my_docs (interactive)
  python rag_chat.py --docs ./my_docs "有給休暇は何日？"  # one question
  python rag_chat.py --docs ./my_docs --show-context   # also print the passages that were found

How a question is answered with --docs:
  1. every file is cut into passages of about --chunk characters (at blank lines, then at line ends);
  2. passages and the question are embedded with a small multilingual model (intfloat/multilingual-e5-small, Japanese and
     English) and the --k most similar passages are taken;
  3. the model gets the recommended system prompt plus "answer only from the documents", the passages numbered [1]..[k] with
     their file names, and the question. This is the format the model was checked with (it uses the documents over its own
     beliefs, says when the answer is absent, ignores instructions written inside a document).
"""
import argparse, glob, os, platform, re, sys

SYSTEM = """あなたは誠実で役に立つAIアシスタントです。
- ユーザーと同じ言語で答えてください。ただし、システムの指示で言語が指定されている場合はそれに従ってください。
- 武器・爆発物・危険な薬物の作り方、人を傷つける方法、犯罪の具体的な手順は教えず、「その質問にはお答えできません」と短く理由を添えて断ってください。危険のない日常的な質問には普通に答えてください。
- 知らないことや確信が持てないことは、作り話をせず「わかりません」と答えてください。資料が与えられた場合は資料の内容を優先し、資料の記述から計算や推論で導けることは答えてください。
- ユーザーが出力の形式や長さを指定したときは、それを厳密に守ってください。指定がなければ普通の文章で答えてください。
- 同じ文を繰り返さず、簡潔に答えてください。"""
RAG = "以下の資料だけに基づいて質問に答えてください。資料に答えがない場合は「資料には記載がありません」と答えてください。"

ap = argparse.ArgumentParser(description="Chat / answer from documents with Qwen3-1.7B-4bit-DWQ")
ap.add_argument("question", nargs="?", help="one question (omit for an interactive session)")
ap.add_argument("--docs", help="a folder of .txt / .md files to answer from")
ap.add_argument("--k", type=int, default=4, help="passages given to the model"); ap.add_argument("--chunk", type=int, default=400)
ap.add_argument("--model", default="baya1116/Qwen3-1.7B-4bit-DWQ")
ap.add_argument("--embedder", default="intfloat/multilingual-e5-small")
ap.add_argument("--backend", choices=["auto", "mlx", "torch"], default="auto")
ap.add_argument("--max-new", type=int, default=768); ap.add_argument("--think", action="store_true", help="let the model think first (slower, better at math)")
ap.add_argument("--show-context", action="store_true")
a = ap.parse_args()

backend = a.backend
if backend == "auto":
    try:
        import mlx_lm  # noqa: F401
        backend = "mlx" if platform.system() == "Darwin" else "torch"
    except ImportError: backend = "torch"


# ---------------- retrieval
def passages(folder, size):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True) + glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)):
        text = open(f, encoding="utf-8", errors="ignore").read(); name = os.path.relpath(f, folder); cur = ""
        for block in re.split(r"\n\s*\n", text):
            for line in block.splitlines() or [""]:
                if cur and len(cur) + len(line) > size: out.append((name, cur.strip())); cur = ""
                cur += line + "\n"
            if len(cur) > size // 2: out.append((name, cur.strip())); cur = ""
        if cur.strip(): out.append((name, cur.strip()))
    return [p for p in out if p[1]]


class Retriever:
    def __init__(self, folder):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch; self.tok = AutoTokenizer.from_pretrained(a.embedder); self.m = AutoModel.from_pretrained(a.embedder).eval()
        self.P = passages(folder, a.chunk)
        if not self.P: sys.exit(f"no .txt / .md text found in {folder}")
        self.E = self.embed([f"passage: {t}" for _, t in self.P])
        print(f"[rag] {len(self.P)} passages from {len({n for n, _ in self.P})} files", file=sys.stderr)

    def embed(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                x = self.tok(texts[i:i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt")
                h = self.m(**x).last_hidden_state; mk = x["attention_mask"][..., None].float()
                out.append(self.torch.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1))
        return self.torch.cat(out)

    def top(self, q, k):
        s = self.E @ self.embed([f"query: {q}"])[0]
        return [self.P[i] for i in s.topk(min(k, len(self.P))).indices.tolist()]


def messages(q, history, found):
    if found is None: return [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": q}]
    docs = "\n\n".join(f"[{i + 1}] ({name}) {t}" for i, (name, t) in enumerate(found))
    return [{"role": "system", "content": SYSTEM + "\n\n" + RAG}] + history + [{"role": "user", "content": docs + "\n\n質問：" + q}]


# ---------------- generation
if backend == "mlx":
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    model, tok = load(a.model)

    def answer(msgs):
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=a.think, tokenize=False)
        smp = make_sampler(temp=0.6, top_p=0.95, top_k=20) if a.think else make_sampler(temp=0.0)
        return generate(model, tok, p, max_tokens=a.max_new, sampler=smp, logits_processors=make_logits_processors(repetition_penalty=1.05))
else:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float32 if dev == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=dt).to(dev).eval()
    d = torch.load(hf_hub_download(a.model, "qwen3-1.7b-4bit-dwq.pt"), map_location="cpu"); g = d["group"]
    for k, v in d.items():   # the 4-bit codes, expanded into the 16-bit model (same numbers as a 4-bit kernel)
        if not isinstance(v, dict): continue
        w = model.get_submodule(k).weight; p = v["q"]; q = torch.stack([p & 15, p >> 4], 2).view(p.size(0), -1).float()
        w.data.copy_((q * v["s"].float().repeat_interleave(g, 1) + v["b"].float().repeat_interleave(g, 1)).to(w.dtype))
    del d
    if dev == "cpu": torch.set_num_threads(os.cpu_count() or 4)

    def answer(msgs):
        x = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=a.think, return_tensors="pt", return_dict=True).to(dev)
        kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if a.think else dict(do_sample=False)
        with torch.no_grad(): y = model.generate(**x, max_new_tokens=a.max_new, repetition_penalty=1.05, **kw)
        return tok.decode(y[0, x["input_ids"].size(1):], skip_special_tokens=True)


def clean(t): return t.split("</think>")[-1].strip()


R = Retriever(a.docs) if a.docs else None
print(f"[rag] backend {backend}, model {a.model}" + (f", answering from {a.docs}" if R else ""), file=sys.stderr)
history = []
while True:
    q = a.question or input("\n> ").strip()
    if not q: break
    found = R.top(q, a.k) if R else None
    if found and a.show_context:
        for i, (n, t) in enumerate(found): print(f"[{i + 1}] ({n}) {t[:200]}", file=sys.stderr)
    out = clean(answer(messages(q, history, found)))
    print(out)
    if a.question: break
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": out}]   # earlier turns kept without their passages
    history = history[-8:]
