"""The search environment of the app (release/USAGE.md section 5) over Wikipedia's API, shared by tools/dc_eval.py and
tools/qwen_search_eval.py. <search>query</search> -> the top English Wikipedia page as "Title: text", served 256 tokens at a
time ("<more>" for the next slice); the same page found again serves its next slice; MAXS searches, MAXM slices per page.
Pages are cached on disk (one request per new query / page; a generic client name, no personal details)."""
import json, os, re, ssl, time, urllib.parse, urllib.request

UA = "Mozilla/5.0 (compatible; eval-script/1.0)"
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or ("/root/.ccr/ca-bundle.crt" if os.path.exists("/root/.ccr/ca-bundle.crt") else None))


class Unavailable(Exception): pass


def _get(url, tries=6):   # Wikipedia answers 429 now and then from shared addresses: back off; never cache a failure
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20, context=CTX) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(min(10 * 2 ** t, 120))
    raise Unavailable(url)


class WikiEnv:
    def __init__(self, tok, cache_path, page_step=256, maxs=5, maxm=8):
        self.tok, self.step, self.maxs, self.maxm = tok, page_step, maxs, maxm
        self.cache_path = cache_path; self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.reset()

    def reset(self): self.failed = False; self.ns = 0; self.page = None; self.served = {}; self.nm = 0; self.slices = []

    def _save(self): json.dump(self.cache, open(self.cache_path, "w"), ensure_ascii=False)

    def lookup(self, q):
        key = "q:" + q
        if key not in self.cache:
            r = _get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": 1}))
            hits = r.get("query", {}).get("search", [])
            self.cache[key] = hits[0]["title"] if hits else None; self._save(); time.sleep(0.5)
        t = self.cache[key]
        if t is None: return None
        pk = "p:" + t
        if pk not in self.cache:
            r = _get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": t, "format": "json", "redirects": 1}))
            pages = r.get("query", {}).get("pages", {})
            txt = next(iter(pages.values()), {}).get("extract", "") if pages else ""
            self.cache[pk] = re.sub(r"\n{2,}", "\n", txt)[:12000]; self._save(); time.sleep(0.5)
        return t, f"{t}: {self.cache[pk]}"

    def _slice(self, title, text):
        ids = self.tok.encode(text, add_special_tokens=False); i = self.served.get(title, 0)
        if i * self.step >= len(ids): return None
        self.served[title] = i + 1; self.page = (title, text); self.slices.append(title)
        return self.tok.decode(ids[i * self.step:(i + 1) * self.step])

    def search(self, q):
        self.ns += 1
        if self.ns > self.maxs: return "\n<information>(no searches left - answer from what you have read)</information>\n"
        q = q.split("||")[0].strip()
        try: hit = self.lookup(q) if q else None
        except Unavailable: self.failed = True; return "\n<information>(no results)</information>\n"
        if not hit: return "\n<information>(no results)</information>\n"
        ch = self._slice(*hit); self.nm = 0
        if ch is None: return "\n<information>(this page is used up - search a different query or answer)</information>\n"
        return f"\n<information>\n{ch}\n[READER] (no extraction)\n</information>\n"

    def more(self):
        self.nm += 1
        if not self.page or self.nm > self.maxm: return "\n<information>(no more of this page - search again or answer)</information>\n"
        ch = self._slice(*self.page)
        if ch is None: return "\n<information>(no more of this page - search again or answer)</information>\n"
        return f"\n<information>\n{ch}\n[READER] (no extraction)\n</information>\n"

    def grounded(self, gold):
        return any(gold.lower() in self.cache.get("p:" + t, "").lower() for t in set(self.slices))
