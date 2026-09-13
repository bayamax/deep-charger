# 指示書 v2: 圧縮コンテキスト付き検索エージェントを iPhone アプリに載せる

対象: ローカルの Claude (Mac 上で作業)。前提知識は「エッジ端末で LLM を動かしたことがある」「検索ループの
初期版 (docs/iphone_agent_spec.md) は作った」まで。それ以降にサーバ側でやったことは全部ここに書く。

---

## 0. 今回の成果物

1. アプリの 4bit モデルを新版 `sft_s1_mlx4` に差し替える (§2)
2. **圧縮コンテキスト (プーラー)** をアプリの生成ループに組み込む (§3)。これが今回の本体
3. 検索環境を現行仕様に合わせる (§4)。初期版との差分は小さい
4. **翻訳レイヤ**: 入力をユーザーの言語から英語へ、思考と検索は英語、最終回答をユーザーの言語へ (§5)
5. Mac 上の Python 参照実装と、iPhone の出力が同じ質問で同じ形式・同じ桁の正答率になること (§6)

---

## 1. これは何か

- モデル: DeepSeek-R1-Distill-Qwen-1.5B 系統 (hidden 1536、28 ブロック)。SFT のあと、検索環境で GRPO を
  200 ステップ回した「grpo_pool3 step200」。プロンプトに system は入れない。
- **圧縮コンテキスト**: 生成中のテキストは全部を KV に残さない。直近 768 トークンだけを生のまま持ち、それより
  古いトークンは「プーラー」という小さな注意ネットが読んで **32 本のベクトル (SP)** に圧縮する。モデルには
  `[質問] + [SP 32 本] + [直近 768 トークン]` だけを見せる。KV キャッシュは 128 トークン生成するごと (と情報
  ブロックを差し込むごと) に質問部分まで切り戻して組み直す。だから端末のメモリと計算量が有界になる。
- 検索: `<search>語 || 知りたいこと</search>` で英語 Wikipedia を引き、`<information>` ブロックとして
  差し込む。`<more/>` で同じページの続き。回数上限あり。`</think>` の後に "answer is …" で終える。
- 成績 (保留 150 問 × 2 回 = 300 本、温度 0.9、この圧縮と検索込み、対比較):

| 構成 | 正答率 | 根拠あり | 着地 | bf16 との差 |
|---|---|---|---|---|
| bf16 (float のまま) | 39.7% | 64% | 98% | - |
| 4bit そのまま (現行のアプリ相当) | 28.0% | 53% | 98% | −11.7 ± 3.0 |
| **4bit + 自己トレース学習 (s1、今回入れるもの)** | **42.3%** | 64% | 98% | +2.7 ± 3.4 |

  アプリ既定の温度 0.6 では bf16 45.7%、4bit そのまま 37.0% (s1 は未測定)。
- s1 は「4bit のまま、自分が GRPO 中に出した正解トレース 1431 本を、上の圧縮コンテキスト越しに再現する」学習。
  動いたのは量子化テンソルのスケールとバイアスだけ。プーラーとトークナイザは不変。詳細は
  docs/quantization_4bit.md。

---

## 2. 在処と、モデルの差し替え

Hugging Face (private): `baya1116/hypernet-sp-distill`。トークンは環境変数 `HF_TOKEN` で渡す。ソースやリポジトリに書かない。

| もの | パス | 備考 |
|---|---|---|
| **新 4bit モデル** | `pooler_distill/grpo_pool3_step200_q4/sft_s1_mlx4/` | model.safetensors 1.0 GB (735 テンソル、198 が量子化)、config.json、generation_config.json、tokenizer.json、tokenizer_config.json、special_tokens_map.json |
| **プーラー (評価に使ったもの)** | `fft_out/pooler.pt` | torch の state_dict (302 MB、fp32、キー `query`, `blocks.{0,1,2}.*`, `ln_out.*`, `out_scale`)。§3-1 参照 |
| プーラーの MLX 実装 | `pooler_mlx.py` (リポジトリ直下) | `PoolerMLX(path)`: `.pt` を読んで forward / forward_with_mass。Swift 移植の正 |
| プーラーの torch 実装 | `box_recover/scripts/pooler_torch.py` | 同じものの torch 版。数式を確認したいとき |
| 参照実装の元 | deep-charger の `ctl/pool_eval.py` | 42.3% を測った評価器そのもの。§3-2 のループはここの `rollout()` を写したもの |
| bf16 の元モデル (参照のみ) | `pooler_distill/grpo_pool3_step200/model/` | |
| 学習パラメータ (参照のみ) | `pooler_distill/grpo_pool3_step200_q4/sft_s1_params.pt` | |
| 詰め込みと検証 (参照のみ) | deep-charger の `ctl/packmlx.py`, `ctl/checkmlx.py` | |

形式: MLX affine 4bit、group_size 64。config.json に `"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}`
と同内容の `quantization_config`、`"torch_dtype": "float16"`。量子化テンソルは全 28 ブロックの
q/k/v/o_proj と gate/up/down_proj、embed_tokens、lm_head の 198 個で、`<name>.weight` (uint32 に 8 個ずつ
詰めた 4bit コード)、`<name>.scales`、`<name>.biases` (fp16、形 [rows, in/64])。それ以外は fp16。
tie_word_embeddings は false。ローダー (mlx-swift 0.29.1) は `.scales` があるモジュールだけ量子化して読む。

差し替え手順:
1. `hf download baya1116/hypernet-sp-distill --include "pooler_distill/grpo_pool3_step200_q4/sft_s1_mlx4/*" --local-dir <作業dir>`
2. 現行のモデルディレクトリと新版で、`model.safetensors` のキー集合・shape・dtype と config の quantization
   ブロックが一致することを確認する。新版は mlx_lm.convert ではなく自前の詰め込みで作っているので、ここは必ず見る。
   不一致があれば差し替えずに差分を報告して止まる。(現行に model.safetensors.index.json や
   chat_template.jinja があって新版に無い、は許容。チャットテンプレートは tokenizer_config.json に入っている。)
3. 一致したら差し替える。旧版はリネームして残す。サンプリング設定、プロンプト、探索周りは変えない。

---

## 3. 圧縮コンテキスト (今回の本体)

### 3-1. プーラー

構造 (`pooler_torch.py` / `pooler_mlx.py` の通り。全部 fp32 で計算する):

- 入力: 圧縮したい過去トークンの **埋め込み** (モデルの embed_tokens を通した [L, 1536]、L ≤ 384) に
  正弦波位置エンコーディング (標準の sin/cos、L 位置分) を足したもの。
- 学習済みクエリ `query` [32, 1536] から始めて、3 ブロック繰り返す:
  1. `lnq1(q)` を query、`lnk(past)` を key/value に **クロス注意** (8 ヘッド、head 192、packed in_proj [4608,1536] + bias、out_proj [1536,1536] + bias。torch の nn.MultiheadAttention と同じレイアウト)。q += 出力。
     ヘッド平均した注意重み [32, L] を **クエリ方向に合計**した [L] を、3 ブロック分足し合わせたものが各過去トークンの「質量 (mass)」。
  2. `lnq2(q)` で **自己注意** (同じ形の重み)。q += 出力。
  3. `lnq3(q)` → FFN (Linear 1536→2048 相当の `ffn.0`、GELU (erf 版)、`ffn.2`)。q += 出力。
- 出力: `ln_out(q)` を各ベクトルごとに L2 正規化し、`|out_scale|` を掛ける。→ SP [32, 1536]。
- LayerNorm の eps は 1e-5。過去トークンが 0 本のときはクロス注意を飛ばす (q はクエリのまま自己注意と FFN だけ通る)。
- 重みは `fft_out/pooler.pt` (`ck["pooler"]` の state_dict、64 テンソル、`args` に heads は無いので既定の 8)。
  Swift で読むなら Mac で一度 safetensors に書き出す (キーはそのまま)。

### 3-2. 生成ループ (pool_eval.py の rollout() をそのまま移す)

定数: `RW=768` (生の窓)、`MAXD=384` (プーラーに渡す最大トークン数)、`CHUNK=128` (組み直し間隔)、
`GEN=1500` (モデルが生成するトークン上限。差し込んだ情報ブロックは数えない)、`MAXS=5`、`MAXM=8`、
`PAGE_STEP=256`、温度はアプリ既定 0.6 (評価は 0.9)、1 問の実時間上限 600 秒。

状態: `q_ids` (質問のプロンプト、§4-1)、`gen` (生成トークン + 差し込んだ情報ブロックのトークン、全部)、
`kept` (窓から押し出されたトークンのうちプーラーに渡す分)、`absorbed` (押し出し済みの位置)。

```
q_ids を KV に prefill する。この長さを MQ とする (以後 KV はここまで切り戻す)。
loop:                                                   # 「ステップ」= 1 回の組み直し
  c0 = len(gen); R = min(c0, RW); nd = c0 - R            # 窓の外に出た本数 nd
  if nd > absorbed:
      kept += gen[absorbed:nd]; absorbed = nd
      if len(kept) > MAXD:
          mass = pooler.forward_with_mass(embed(kept))   # [len(kept)]
          kept = 質量上位 MAXD 本を、元の順序を保って残す
  spv = pooler.forward(embed(kept))                      # [32, 1536]、kept が空でも 32 本出る
  block = concat(spv, embed(gen[c0-R : c0]))             # [32 + R, 1536]、モデルの dtype (fp16)
  KV を MQ に切り戻し、block を inputs_embeds として位置 MQ .. MQ+len(block)-1 で流す
  last = 最後の logits
  for CHUNK 回:
      nx = sample(last)                                  # §4-3: "<information" を作る候補は弾く
      if nx == EOS: 終了
      gen.append(nx)
      if 直近 8 トークンが全部同じ: 終了 (dead)
      txt = decode(gen)
      if 検索タグが閉じた:  情報ブロックを作り、そのトークンを gen に append して break  # 組み直しへ
      if more タグ:          同上
      if "</think>" が出ていて answer_complete(その後ろ): 終了
      nx を 1 トークン流して last を更新 (位置は続き)
  終了条件が立っていなければ loop の頭へ
```

要点:
- SP ベクトルは **inputs_embeds** としてモデルに入れる。Swift 側では LLMEval の `model(tokens)` ではなく、
  embed_tokens を通した後のテンソルを transformer 本体に渡す口が要る。位置 (RoPE) は MQ からの連番で、
  SP の 32 本も位置を消費する。
- 差し込んだ情報ブロックのトークンは `gen` に入り、やがて窓から押し出されてプーラーに行く (episodic)。
- 組み直しごとの prefill は最大 32 + 768 トークン。128 トークンに 1 回なので、実効コストは
  1 トークンあたり約 6 トークン分の prefill。iPhone 15 Pro で体感が許容か測ること。
- 評価器はモデルを bf16、プーラーを fp32 で走らせた。アプリは fp16 + fp32 になる。
  SP の L2 正規化と out_scale のおかげで値域は小さく、fp16 で問題ないはずだが §6 で確認する。

---

## 4. 検索環境 (初期版 docs/iphone_agent_spec.md からの差分)

### 4-1. プロンプト
初期版と同じ。system 無し、`apply_chat_template([user], add_generation_prompt=True)` の後ろに
`<think>\n` を足す (テンプレートが既に付けていれば足さない)。

### 4-2. 情報ブロック (初期版と同じ文字列。変えない)
- 通常: `"\n<information>\n{chunk}\n[READER] (no extraction)\n</information>\n"`
- 検索語が空: `"\n<information>(no results)</information>\n"`
- 結果なし: chunk = `"(no results)"` を通常形式で
- 検索回数超過 (`ns > MAXS`): `"\n<information>(no searches left - answer from what you have read)</information>\n"`
- more が尽きた: `"\n<information>(no more of this page - search again or answer)</information>\n"`
- **新規** 同じページが使い切られた: `"\n<information>(this page is used up - search a different query or answer)</information>\n"`

### 4-3. 新規: samepage
同じページ (ページ文字列の先頭 120 文字をキーにする) がまた検索でヒットしたら、先頭 256 トークンをもう一度
出すのではなく、**前回の続き** (そのページの現在オフセットから 256 トークン) を出す。続きが無ければ上の
"used up" 通知。`<more/>` で読み進めたオフセットも同じ表に反映する。実装は pool_eval.py の
`seen_pages` / `page_off` / `cur_key` の通り。

### 4-4. サンプリング
温度サンプリング。`"<information"` を生成テキストに作ってしまう候補 (直近 16 トークン + 候補で判定) は
弾いて最大 8 回引き直し、それでもダメなら argmax。`</think>` が直近 40 トークンに出ていても greedy には
しない (plain デコード)。繰り返しペナルティ等は使わない。

### 4-5. Wikipedia API、回答の取り出し、正誤判定
初期版と同一 (UA 必須、search → exintro → 本文、`"{title}: "` 先頭、40000 文字、`answer_complete`、
`head_sentence`、`norm`)。

---

## 5. 翻訳レイヤ (Apple 標準のオンデバイス翻訳)

目的: モデルもインデックスも英語なので、ユーザーが日本語などで聞いても内部は全部英語で回し、最後だけ戻す。

- 言語判定: `NLLanguageRecognizer` で入力の言語を取る。英語なら翻訳レイヤは素通し。
- 翻訳: `Translation` フレームワーク (iOS 17.4+ / macOS 14.4+)。SwiftUI の `.translationTask(configuration)`
  で `TranslationSession` を受け取り、`session.translate(text)` を使う。設定は
  `TranslationSession.Configuration(source: 入力言語, target: .init(identifier: "en"))` と、その逆。
  初回は `LanguageAvailability().status(from:to:)` を見て、未ダウンロードなら `session.prepareTranslation()`
  で言語パックを入れる (ユーザーに一度ダイアログが出る)。ネットワーク不要、端末内で完結。
- 流れ:
  1. 入力 → 英語に翻訳 → その英文を質問としてプロンプトに入れる (翻訳前の原文は入れない)。
  2. 思考・検索・情報ブロック・回答は全部英語のまま。画面のログは英語で出してよい。
  3. `</think>` 後の最終回答 (`head_sentence` で切った 1 文) を、入力言語に翻訳して表示する。
  4. 英語の原文回答も小さく併記する。固有名詞 (人名、作品名) は翻訳で崩れることがあり、正誤の確認は英語側でしかできない。
- 注意: 質問文に含まれる固有名詞が翻訳で英語表記にならないと検索が外れる。まずは日本語 → 英語で
  5 問ほど試し、翻訳後の質問文を見て判断する。ダメな固有名詞が多ければ、翻訳前の入力からカタカナ以外の
  英字列をそのまま残すなどの手当てを検討する (今回は必須ではない)。

---

## 6. 手順と受け入れ条件

1. **Mac 参照実装を先に作る**: Python + mlx-lm で §3-2 のループを書き、`pooler_mlx.py` の `PoolerMLX` と
   新版 `sft_s1_mlx4` で保留問題を 10 問回す。正答率は 40% 前後、検索回数は 1 問あたり 5 回前後、
   `</think>` まで着地するのが 95% 以上、なら合っている。ログ (検索語、返したチャンク、最終回答) を保存する。
   保留問題は HF の `grpo_assets/mus_run/eval_step200/eval_heldout_300q_g2.jsonl` (`q`, `gold`)。
2. iPhone に移植する。§3 のプーラーと組み直し、§4 の samepage、§5 の翻訳。
3. 受け入れ:
   - 同じ質問で Mac と iPhone の出力形式が同じ (`<search>` → `<information>` → … → `</think>` → "answer is …")。
   - 組み直しが 128 トークンごとに起きていて、KV が `MQ + 32 + 768 + 128` を超えない。
   - 10 問の正答数を Mac と iPhone で記録し、桁が合う (10 問なので ±2 は誤差)。
   - 日本語で聞いて日本語の回答が返り、英語原文も見える。
   - 1 トークンあたりの体感速度 (tok/s) と、1 問の所要時間を記録する。

---

## 7. 補足

- プーラーについて未確定点が一つある。42.3% と 39.7% を測った評価器は `fft_out/pooler.pt` を使っていたが、
  この系統の学習 (SFT と GRPO) ではプーラーも一緒に更新されていて、その学習後のプーラー
  (`pooler_distill/grpo_pool3_step200/pooler.safetensors`、キーは prefix 無しで同じ名前) は評価に
  使われていなかった。学習後のプーラーで測り直すと数字が変わる可能性がある。**今回アプリに入れるのは
  評価で 42.3% を出した組み合わせ (sft_s1_mlx4 + fft_out/pooler.pt)** とし、測り直しの結果が出たら
  差し替えを指示する。プーラーの読み込み口はファイルを差し替えるだけで済むように作っておくこと。
- 学習時の完全なソースは HF の `box_recover/scripts/grpo_e2e_torch.py` (`sp_rollout`) と
  deep-charger の `ctl/pool_eval.py`。迷ったら pool_eval.py の `rollout()` を正とする。
