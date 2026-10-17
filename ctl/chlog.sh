#!/bin/sh
# 文字モデル（箱 CH）の様子。  chlog = 要約 / chlog loss = 学習ロスの推移 / chlog chat = チャット評価と返答 / chlog all = 全部
U="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/pooler_distill/chatsft/audit/boxlog_CH.txt"
L=$(curl -sL "$U?t=$(date +%s)") || { echo "取得失敗"; exit 1; }
echo "$L" | head -2
case "$1" in
  loss) echo "$L" | grep -E '^\[step' | sed -E 's/^\[step ([0-9]+)\] [0-9]+ layers: train ([0-9.]+) bpc, lr ([^,]+),.*, ([0-9.]+B) chars,.* ([0-9]+) min$/step \1  train \2  lr \3  \4  \5 min/' ;;
  chat) echo "$L" | grep -E '^\[chat\]|^\[chat-sample\]' | tail -n 12 ;;
  all)  echo "$L" ;;
  *)    echo "--- 検証（低いほど良い） ---"; echo "$L" | grep -E '^\[val\] step [0-9]+ .*min\)' | tail -n 6
        echo "--- チャット評価（GPT-2 small 0.845 / 目標 0.887） ---"; echo "$L" | grep -E '^\[chat\]' | tail -n 4
        echo "--- 学習ロス（初見データ） ---"; echo "$L" | grep -E '^\[step' | tail -n 4
        echo "--- 状態 ---"; echo "$L" | grep -E '^\[cool\]|^\[plateau\]|^\[ctl\]|^\[diverge\]|^\[grow\]|Traceback|CHARLM2_DONE|^\[stop\]|SFT_DONE' | tail -n 6 ;;
esac
