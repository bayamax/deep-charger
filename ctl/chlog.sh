#!/bin/sh
# 文字モデル（箱 CH）の様子。  chlog = 要約 / chlog loss = 学習ロスの推移 / chlog chat = チャット評価と返答 / chlog all = 全部
U="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/pooler_distill/chatsft/audit/boxlog_CH.txt"
L=$(curl -sL "$U?t=$(date +%s)") || { echo "取得失敗"; exit 1; }
echo "$L" | head -2
case "$1" in
  loss) echo "--- 学習ロス（初見データ）: 生の値と EMA（α=0.1、約 50 分のならし） ---"
        echo "$L" | grep -E '^\[step' | awk '{for(i=1;i<=NF;i++){if($i=="train")v=$(i+1); if($i=="min")m=$(i-1)}; gsub(/[^0-9]/,"",$2); e=(n++?0.9*e+0.1*v:v); printf "step %s  %s min  train %.3f  EMA %.3f\n",$2,m,v,e}' ;;
  chat) echo "$L" | grep -E '^\[chat\]|^\[chat-sample\]' | tail -n 12 ;;
  all)  echo "$L" ;;
  *)    echo "--- 検証（低いほど良い） ---"; echo "$L" | grep -E '^\[val\] step [0-9]+ .*min\)' | tail -n 6
        echo "--- チャット評価（GPT-2 small 0.845 / 目標 0.887） ---"; echo "$L" | grep -E '^\[chat\]' | tail -n 4
        echo "--- 学習ロス EMA（α=0.1、約 50 分のならし） ---"; echo "$L" | grep -E '^\[step' | awk '{for(i=1;i<=NF;i++){if($i=="train")v=$(i+1); if($i=="min")m=$(i-1)}; e=(n++?0.9*e+0.1*v:v); if(n%6==0)printf "%s min  EMA %.3f\n",m,e} END{printf "最新 %s min  EMA %.3f（生 %.3f）\n",m,e,v}' | tail -n 6
        echo "--- 状態 ---"; echo "$L" | grep -E '^\[cool\]|^\[plateau\]|^\[ctl\]|^\[diverge\]|^\[grow\]|Traceback|CHARLM2_DONE|^\[stop\]|SFT_DONE' | tail -n 6 ;;
esac
