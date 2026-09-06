"""Minimal stand-in for the lost runtime.web_search module.

The GRPO/SFT harness only calls WIKI._http_get(url) -> dict (parsed JSON), so that is all this
provides: a polite Wikipedia API GET with the project's User-Agent, retries, and maxlag handling.
"""
import json, ssl, time, urllib.request

UA = {"User-Agent": "deep-charger-grpo-ep/1.0 (research; bayamax@icloud.com)"}


class WikipediaSearch:
    def __init__(self, n_results=3, timeout=8.0):
        self.n_results = n_results
        self.timeout = timeout
        self.ctx = ssl.create_default_context()
        try:
            import certifi
            self.ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass

    def _http_get(self, url, tries=3):
        for i in range(tries):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=max(self.timeout, 20), context=self.ctx) as r:
                    out = json.loads(r.read().decode())
                if isinstance(out, dict) and out.get("error", {}).get("code") == "maxlag":
                    time.sleep(5 * (i + 1)); continue
                return out
            except Exception:
                time.sleep(2 * (i + 1))
        return {}
