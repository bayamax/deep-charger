# 指示書: 検索エージェント(ckpt_mus200)を iPhone 単体で動かす

対象: ローカルの Claude(Mac 上で作業)。目的は、GRPO 学習済みモデル(held-out 41.5%)を iPhone に載せ、
iPhone から直接 Wikipedia API を叩く自前ループで学習環境と同じ挙動を再現すること。

---

## 0. 成果物

1. `mus200_merged/` … bf16_mus_pure に LoRA をマージした HF 形式モデル
2. `mus200_mlx4/` … 上記の MLX 4bit 版(iPhone 用)
3. iOS アプリ(mlx-swift-examples の LLMEval を土台にした 1 画面アプリ)
   - 質問を入力 → `<search>` / `<more/>` を検知して Wikipedia API を叩き、`<information>` を差し込んで生成を続ける
   - 往復ログ(検索語、返したチャンク、最終回答)を画面に出す
4. 動作確認ログ(Mac 上の Python 参照実装と iPhone の出力が同じ質問で同じ形式になること)

---

## 1. モデルの用意

### 1-1. LoRA の取得とマージ

```bash
pip install -U transformers peft huggingface_hub torch
hf download baya1116/hypernet-sp-distill --include "grpo_assets/mus_run/ckpt_step200/*" --local-dir hfdl
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE = "<bf16_mus_pure のパス>"          # ローカルにある SFT 原本
ADP  = "hfdl/grpo_assets/mus_run/ckpt_step200"
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16")
m = PeftModel.from_pretrained(base, ADP).merge_and_unload()
m.save_pretrained("mus200_merged")
AutoTokenizer.from_pretrained(BASE).save_pretrained("mus200_merged")
```

LoRA の設定(参考): r=16, alpha=320, target = q/k/v/o/gate/up/down, layers 20–27 のみ。adapter_config.json にそのまま入っている。

### 1-2. MLX 4bit へ変換

```bash
pip install -U mlx-lm
mlx_lm.convert --hf-path mus200_merged --mlx-path mus200_mlx4 -q --q-bits 4 --q-group-size 64
```

### 1-3. Mac で挙動確認(必須)

```bash
mlx_lm.generate --model mus200_mlx4 --max-tokens 300 --temp 0.9 \
  --prompt "Who directed the film Ray Parker Jr is best known for writing and performing the theme song to?"
```

`<think>` の後に `<search>キーワード || 知りたいこと</search>` が出れば OK。
チャットテンプレートは tokenizer_config.json の chat_template をそのまま使う(system プロンプトは無し。後述)。

---

## 2. プロンプトの組み方(学習時と同一にすること)

学習コードは次のように組んでいた。system メッセージは入れない。

```python
head = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                     add_generation_prompt=True, tokenize=False)
if not head.rstrip().endswith("<think>"):
    head += "<think>\n"
prompt_ids = tokenizer.encode(head)
```

つまり `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n` から生成を始める。
モデルは `<think>` 内で検索を繰り返し、`</think>` の後に最終回答文を書く。

---

## 3. 生成ループの仕様(学習環境 grpo_ep_more.py の episode() を移植)

### 定数

| 名前 | 値 | 意味 |
|---|---|---|
| MAXS | 5 | 1 問あたりの検索回数上限 |
| MAXM | 8 | 1 問あたりの `<more/>` 回数上限 |
| GEN | 1500 | 生成トークン上限(情報ブロックは数えない) |
| TEMP | 0.9 | サンプリング温度(体感用なら 0.6 程度でもよい) |
| PAGE_STEP | 256 | 1 回に見せるページの塊(トークン数) |

### 正規表現・固定文字列(そのまま使う)

```python
CLOSE  = re.compile(r"<search>(.*?)</\s*search\s*[^\w<]{0,3}$", re.S)   # 生成テキスト末尾で閉じた search
MORE   = re.compile(r"<\s*/?\s*more\s*/?\s*>\s*$", re.I)               # 生成テキスト末尾の more タグ
NOTICE = "(no searches left - answer from what you have read)"
NOMORE = "(no more of this page - search again or answer)"
NORES  = "(no results)"
TAG    = "<information"   # モデル自身に <information を書かせない(生成時に禁止トークン扱い)
```

### 1 トークンごとの処理(擬似コード)

```
txt = これまでに生成したテキスト(情報ブロック込み)
tok = sample(logits, temp)          # "<information" を作る候補は弾いて再サンプル(最大 8 回、最後は argmax)
if tok == EOS: break
txt += decode(tok)

# --- 検索 ---
si = txt.rfind("<search>")
m  = CLOSE.search(txt, si) if si >= 0 else None
if m and (これまでに処理した search の数 == 出現した <search> の数 - 1):
    ns += 1
    body = m.group(1).strip()
    kw, ask = body.split("||", 1) を strip したもの   # "||" が無ければ kw = ask = body
    if kw == "":            blk = "\n<information>(no results)</information>\n"
    elif ns > MAXS:         blk = f"\n<information>{NOTICE}</information>\n"
    else:
        chunk, page_ids = serve(kw)        # 下記
        if page_ids は空:   blk = "\n<information>\n(no results)\n[READER] (no extraction)\n</information>\n"
        else:
            page_off = PAGE_STEP; nm は据え置き
            blk = f"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"
    txt += blk;  blk をトークン化してモデルに食わせる(KV キャッシュに追加);  continue

# --- more ---
if MORE.search(txt):
    nxt = page_ids[page_off : page_off+PAGE_STEP] if nm < MAXM else []
    if nxt が空:  blk = f"\n<information>{NOMORE}</information>\n"
    else:
        nm += 1; page_off += PAGE_STEP
        blk = f"\n<information>\n{decode(nxt)}\n[READER] (no extraction)\n</information>\n"
    txt += blk; モデルに食わせる; continue

# --- 終了判定 ---
if "</think>" in txt and answer_complete(txt.split("</think>")[-1]): break
```

注意: 学習時は `(no results)` の時も `[READER]` 行付きの形式ではなく、
`serve()` が `"(no results)"` を chunk として返し `"\n<information>\n(no results)\n[READER] (no extraction)\n</information>\n"` になっていた。上の擬似コードはその通り。
`kw` が空文字のときだけ `\n<information>(no results)</information>\n`(改行なし形式)になる。

### serve(kw): Wikipedia API

学習時の fetch() と同じ順序で叩く。User-Agent は必ず付ける(付けないと 403 になることがある)。

```
UA = "deep-charger-grpo-ep/1.0 (research; bayamax@icloud.com)"
API = https://en.wikipedia.org/w/api.php  (すべて format=json, maxlag=5 を付与)

1) 検索:      action=query&list=search&srsearch={kw}&srlimit=3
   → hits = 上位 3 件の title。0 件なら "" を返す
2) intro:     action=query&prop=extracts&exintro=1&explaintext=1&exlimit=max&redirects=1&titles={hits を | 連結}
   → extract が空でないものを pages に入れる
3) 本文:      hits の順に、pages にあるものについて
              action=query&prop=extracts&explaintext=1&redirects=1&titles={title}
   → extract があれば  page = f"{title}: {extract[:40000]}" を返す(1 件目で確定)
   → 無ければ  page = f"{title}: {pages[title]}"(intro だけ)

page_ids = tokenizer.encode(page)(特殊トークンなし)
chunk    = decode(page_ids[:PAGE_STEP])
```

ページ文字列の先頭が `"{title}: "` で始まる点、40000 文字で切る点、
explaintext の見出しが `== Section ==` 形式で入る点を変えないこと(モデルはこの形式で学習している)。

### 回答の取り出し

```python
ABBR = {"st","mr","mrs","ms","dr","jr","sr","mt","vs","no","inc","ltd","co"}

def answer_complete(seg):            # seg = "</think>" より後ろ
    m = re.search(r"answer is (.+)", seg, re.I | re.S)
    if not m: return False
    a = m.group(1)
    for mm in re.finditer(r"[.!?](?=\s|$)", a):
        t = re.split(r"[\s(\"]", a[:mm.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR: continue   # "J." や "Dr." では終わらない
        return True
    return False

def head_sentence(a):                # 最終回答の 1 文目(120 文字まで)
    h = a.strip().split("\n")[0]
    for m in re.finditer(r"[.!?](?=\s|$)", h):
        t = re.split(r"[\s(\"]", h[:m.start()])[-1]
        if len(t) == 1 or t.lower() in ABBR: continue
        return h[:m.end()][:120]
    return h[:120]

final_answer = head_sentence(txt.split("</think>")[-1].strip())
```

正誤判定(自分で確認したい時): `norm(x) = re.sub(r"[^a-z0-9 ]"," ",x.lower()).strip()` で正規化し、
`" "+norm(gold)+" " in " "+norm(final_answer)+" "` なら正解。

---

## 4. Mac 上の参照実装(先に作る)

上記ループを Python + mlx-lm で 150 行程度に書き、`mus200_mlx4` で 5 問ほど回して往復ログを保存する。
これが iPhone 実装の正解データになる。mlx-lm では 1 トークンずつ生成しながら KV キャッシュを保持できる
(`mlx_lm.generate` の stream_generate か、`mlx_lm.models` の cache を直接使う)。
情報ブロックの差し込みは「ブロックをトークン化して cache に流し込む(logits は捨てる)」で実現する。

停止語で簡略化する場合: `</search>`、`<more/>`、`<more>`、`</more>` を停止語にして生成を止め、
返ってきたテキストに上の正規表現をかける。学習時は 1 トークンごとに判定していたが、体感用ならこれで十分。

---

## 5. iOS アプリ

- 土台: https://github.com/ml-explore/mlx-swift-examples の `Applications/LLMEval`
- モデル: `mus200_mlx4` をアプリの Bundle に同梱するか、初回起動時に HF からダウンロード
  (HF へ上げる場合: `hf upload baya1116/hypernet-sp-distill mus200_mlx4 mlx/mus200_mlx4`)
- 変更点:
  1. 生成を `TokenIterator` で 1 トークンずつ回し、上記ループを Swift で実装(正規表現は NSRegularExpression で同等に)
  2. Wikipedia API は `URLSession` で同期的に待つ(async/await)。User-Agent ヘッダを必ず付ける
  3. 情報ブロックの差し込みは、ブロックをトークン化して同じ KV キャッシュに `prefill` する
     (LLMEval の `generate` を分解して cache を保持する形にする)
  4. 画面: 質問入力、進行中のテキスト(検索語と情報ブロックを色分け)、最終回答、検索回数 / more 回数、tok/s
- 対象端末: iPhone 15 Pro 以降推奨(1.5B 4bit で 20〜30 tok/s、メモリ 1.2GB 程度)
- 署名: 無料の Apple ID で 7 日間のサイドロード可

---

## 6. 受け入れ条件

1. Mac 参照実装と iPhone で同じ質問を投げ、`<search>` → `<information>` → … → `</think>` → 回答、という形式が両方で出る
2. 検索が 5 回を超えたら NOTICE、`<more/>` が尽きたら NOMORE が入る
3. 回答が "answer is …" で終わったら止まる
4. 往復ログを 5 問分保存し、うち正解数を記録する(held-out の目安は 40% 前後)

---

## 7. 補足(環境の背景)

- 学習時のページ取得はこの API と同じもので、箱側で 6000 ページのキャッシュを使っていただけ。検索順位が日によって変わる以外は同じ挙動になる。
- `[READER] (no extraction)` は司書(抽出器)を外した時の固定文字列で、モデルはこの行を見慣れている。消さないこと。
- 質問データや評価スクリプト、学習時の完全なソース(grpo_ep_more.py)は HF の `grpo_assets/box_archive_20260904/` にある。迷ったらそこの episode() と serve() を正とする。
