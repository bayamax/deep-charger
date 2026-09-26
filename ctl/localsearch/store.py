"""Read side of the article store: (block, position) -> (title, text), one zstd frame per block."""
import collections, json, os, struct

import zstandard as zstd

IDX = struct.Struct("<QIH")


class Store:
    def __init__(self, path, cache_blocks=64):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.block = self.meta["block"]
        raw = open(os.path.join(path, "blocks.idx"), "rb").read()
        self.blocks = [IDX.unpack_from(raw, i) for i in range(0, len(raw), IDX.size)]   # (offset, length, count)
        self.n_docs = sum(b[2] for b in self.blocks)
        self.f = open(os.path.join(path, "docs.bin"), "rb")
        self.d = zstd.ZstdDecompressor()
        self.cache = collections.OrderedDict()
        self.cache_blocks = cache_blocks
        self._titles = None

    def _block(self, b):
        if b in self.cache:
            self.cache.move_to_end(b)
            return self.cache[b]
        off, ln, _ = self.blocks[b]
        self.f.seek(off)
        docs = [x.split(b"\n", 1) for x in self.d.decompress(self.f.read(ln)).split(b"\n\x00") if x]   # "title\nbody" records
        self.cache[b] = docs
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return docs

    def doc(self, i):
        """title, text of article i (store order). Blocks are full except possibly the last of each shard."""
        b, r = self._locate(i)
        t, body = self._block(b)[r]
        return t.decode("utf-8"), body.decode("utf-8").rstrip("\n")

    def _locate(self, i):
        # blocks are almost all of size self.block; walk from the estimate to be exact
        b = min(i // self.block, len(self.blocks) - 1)
        start = self._starts()[b]
        while start > i:
            b -= 1; start = self._starts()[b]
        while b + 1 < len(self.blocks) and self._starts()[b + 1] <= i:
            b += 1; start = self._starts()[b]
        return b, i - start

    def _starts(self):
        if not hasattr(self, "_st"):
            s, acc = [], 0
            for _, _, c in self.blocks:
                s.append(acc); acc += c
            self._st = s
        return self._st

    def titles(self):
        if self._titles is None:
            self._titles = open(os.path.join(self.path, "titles.txt"), encoding="utf-8").read().split("\n")
            if self._titles and self._titles[-1] == "":
                self._titles.pop()
        return self._titles
