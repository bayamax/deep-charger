"""Reward for coding problems (the user: see whether a checker can be built here, and if it works, merge coding problems in).

A coding problem's gold is {"tests": [{"input": ..., "output": ...}, ...], "tl": seconds}. The answer is right when the last
```python block after </think> prints the expected output on every test (whitespace-insensitive tokens, floats to 1e-6).
The program runs in a fresh temporary directory as the unprivileged user nobody when we are root (it cannot read /root:
the hub token, the weights), with an empty environment, Python's isolated mode, and limits on CPU time, memory, file size
and processes; it stops at the first failing test. Results are cached per (code, tests).

  python sp_codecheck.py selftest <problems.jsonl>   the reference solutions (field "ref") must pass, and a wrong program fail
"""
import hashlib, json, os, re, resource, shutil, subprocess, sys, tempfile

_CACHE = {}
_PY = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
_BLOCK = re.compile(r"```[ \t]*(?:python3?|py|Python3?)?[ \t]*\n(.*?)```", re.S)


def extract(text):
    ans = text.split("</think>")[-1]
    b = _BLOCK.findall(ans)
    return b[-1] if b else None


def _limits(cpu):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 24, 1 << 24))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()
        if os.getuid() == 0:
            os.setgroups([]); os.setgid(65534); os.setuid(65534)
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    return f


def _same(out, exp):
    a, b = out.split(), exp.split()
    if a == b: return True
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if x == y: continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fy)): return False
    return True


def run_tests(code, tests, tl=4):
    d = tempfile.mkdtemp(prefix="cc_")
    try:
        os.chmod(d, 0o777)
        p = os.path.join(d, "sol.py")
        with open(p, "w") as f: f.write(code)
        os.chmod(p, 0o644)
        for t in tests:
            try:
                r = subprocess.run([_PY, "-I", "sol.py"], input=t["input"].encode(),
                                   capture_output=True, timeout=tl * 1.5 + 1, cwd=d, env={"PATH": "/usr/bin:/bin", "HOME": d},
                                   preexec_fn=_limits(int(tl) + 1))
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0 or not _same(r.stdout.decode(errors="replace"), t["output"]): return False
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)


def code_correct(text, gold):
    code = extract(text)
    if not code: return False
    k = (hashlib.sha1(code.encode()).hexdigest(), hashlib.sha1(json.dumps(gold, sort_keys=True).encode()).hexdigest())
    if k not in _CACHE: _CACHE[k] = run_tests(code, gold["tests"], gold.get("tl", 4))
    return _CACHE[k]


if __name__ == "__main__" and sys.argv[1] == "sandbox":   # on a box: a right program passes, a wrong one fails, /root is unreadable
    t = [{"input": "3 4\n", "output": "7\n"}]
    print("[cc] sandbox: right", run_tests("a, b = map(int, input().split())\nprint(a + b)", t),
          "| wrong", run_tests("print(8)", t),
          "| reads /root", run_tests("import os\nos.listdir('/root')\nprint(7)", t) or
          run_tests("open('/root/.hf_token').read()\nprint(7)", t), "| sees HF_TOKEN", run_tests("import os\nprint(7 if 'HF_TOKEN' in os.environ else 0)", t),
          "| uid", os.getuid(), "| python", _PY)
if __name__ == "__main__" and sys.argv[1] == "selftest":
    import time
    rows = [json.loads(l) for l in open(sys.argv[2])]
    ok = bad = 0; t0 = time.time()
    for r in rows:
        ref = "<think>x</think>\n```python\n" + r["ref"] + "\n```"
        ok += code_correct(ref, r["gold"])
        bad += code_correct("<think>x</think>\n```python\nprint(42)\n```", r["gold"])
    print(f"reference solutions pass {ok}/{len(rows)}, a wrong program passes {bad}/{len(rows)}, {time.time() - t0:.1f}s")
