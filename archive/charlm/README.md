# charlm（アーカイブ）

文字単位の小さな言語モデル（5000 万パラメータ、会話とゲームの行動選び）の実験の跡です。deep-charger 本体とは関係ありません。

- 整理したコードとモデルは別の場所にあります：
  - コード（と 4bit 版モデル）：https://github.com/bayamax/charlm-agent
  - モデル：https://huggingface.co/baya1116/charlm-agent
- `code/`：実験当時のコード（学習・データ作り・採点・村のシミュレーション）
- `boxes/`：Vast.ai の箱で実験を回したときのスクリプト。データやモデルの取得先（`.../charlm/...`）は、今は Hugging Face の `baya1116/hypernet-sp-distill` の `archive/charlm/` にあります。
