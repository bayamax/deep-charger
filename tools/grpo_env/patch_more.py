"""Derive grpo_ep_more.py from grpo_ep_torch.py: wire the <more/> tag the policy
already emits to the next PAGE_STEP slice of the page it last read. Asserts on every
edit, so a changed source fails loudly instead of producing a half-patched trainer."""
import ast
S = open("/root/work/grpo_ep_torch.py").read()
def sub(old, new, tag):
    global S
    assert old in S, "MISSING: " + tag
    S = S.replace(old, new, 1)

sub('MAXS = int(os.environ.get("GRPO_MAXS", "5"))',
    'MAXS = int(os.environ.get("GRPO_MAXS", "5"))\n'
    'MAXM = int(os.environ.get("GRPO_MAXM", "8"))   # <more/> continuations per episode', "MAXM")

sub('''def serve(kw, ask):
    """Four-arm test settled it: the page's head chunk, no reader aiming."""
    pg = get_page(kw)
    if not pg:
        return "(no results)", "(no extraction)", ""
    ids = tokenizer.encode(pg, add_special_tokens=False)
    return tokenizer.decode(ids[:PAGE_STEP]), "(no extraction)", pg''',
'''def serve(kw, ask):
    """Four-arm test settled it: the page's head chunk, no reader aiming.

    The page's token ids come back too: <more/> walks them one PAGE_STEP at a time,
    so evidence below the head chunk is reachable without spending a search.
    """
    pg = get_page(kw)
    if not pg:
        return "(no results)", "(no extraction)", "", []
    ids = tokenizer.encode(pg, add_special_tokens=False)
    return tokenizer.decode(ids[:PAGE_STEP]), "(no extraction)", pg, ids''', "serve")

sub('CLOSE = re.compile(r"<search>(.*?)</\\s*search\\s*[^\\w<]{0,3}$", re.S)',
'''CLOSE = re.compile(r"<search>(.*?)</\\s*search\\s*[^\\w<]{0,3}$", re.S)
# The policy already emits these unprompted (<more/> 257x, <more> 19x in the s100 run)
# and until now nothing came back. Counting occurrences, rather than anchoring to the
# end of the text, is what the search branch does and is what works: a token often
# carries the closing ">" together with the punctuation that follows it, so the text
# never ends on the bare tag.
MORE = re.compile(r"<\\s*/?\\s*more\\s*/?\\s*>", re.I)
NOMORE = "(no more of this page - search again or answer)"''', "MORE regex")

sub('    txt, ns, served, hints, queries_out, page_gold = "", 0, [], [], [], []\n    n_gen = 0',
    '    txt, ns, served, hints, queries_out, page_gold = "", 0, [], [], [], []\n'
    '    nm, nmt, page_ids, page_off = 0, 0, [], 0\n    n_gen = 0', "episode state")

sub('''                chunk, span, page_full = serve(kw, ask)
                served.append(chunk)
                hints.append(span)
                page_gold.append(has(page_full, gold))''',
'''                chunk, span, page_full, pids = serve(kw, ask)
                served.append(chunk)
                hints.append(span)
                page_gold.append(has(page_full, gold))
                page_ids, page_off = pids, PAGE_STEP''', "serve call")

sub('''            npos += len(new)
            continue
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):''',
'''            npos += len(new)
            continue
        if len(MORE.findall(txt)) > nmt:
            nmt += 1
            nxt = page_ids[page_off:page_off + PAGE_STEP] if nm < MAXM else []
            if not nxt:
                blk = f"\\n<information>{NOMORE}</information>\\n"
            else:
                nm += 1
                page_off += PAGE_STEP
                chunk = tokenizer.decode(nxt)
                served.append(chunk)
                hints.append("(no extraction)")
                blk = f"\\n<information>\\n{chunk}\\n[READER] (no extraction)\\n</information>\\n"
            new = tokenizer.encode(blk, add_special_tokens=False)
            ids += new
            mask += [0] * len(new)
            narr = torch.tensor([new], device=DEV)
            npo = torch.arange(npos, npos + len(new), device=DEV)
            out = model(input_ids=narr, past_key_values=past, position_ids=npo.unsqueeze(0),
                        cache_position=npo, use_cache=True)
            logits = out.logits[0, -1, :]
            npos += len(new)
            continue
        if "</think>" in txt and answer_complete(txt.split("</think>")[-1]):''', "more branch")

sub('''    return ids, mask, (ns, grounded, landed, correct, bridge), \\
        {"queries": queries_out, "answer": (ans if landed else ""), "hints": hints[:5],
         "text": tokenizer.decode(ids[len(prompt_ids):])}''',
'''    return ids, mask, (ns, grounded, landed, correct, bridge, nm), \\
        {"queries": queries_out, "answer": (ans if landed else ""), "hints": hints[:5],
         "more": nm, "text": tokenizer.decode(ids[len(prompt_ids):])}''', "episode return")

sub('    return [1.0 if (g and c) else 0.0 for _, g, _, c, _ in group]',
    '    return [1.0 if (g and c) else 0.0 for _, g, _, c, _, _ in group]', "price")

sub('"correct": inf[3], "trusted": inf[4],',
    '"correct": inf[3], "trusted": inf[4], "more": inf[5],', "rollout record")

sub('''            f"srch={sum(i[0] for i in step_info)/n:.1f} |grad|={gnorm:.4f} "''',
'''            f"srch={sum(i[0] for i in step_info)/n:.1f} "
            f"more={sum(i[5] for i in step_info)/n:.2f} "
            f"morerate={sum(1 for i in step_info if i[5])/n:.0%} |grad|={gnorm:.4f} "''', "step log")

ast.parse(S)
open("/root/work/grpo_ep_more.py", "w").write(S)
print("PATCH OK ->", len(S.splitlines()), "lines")
