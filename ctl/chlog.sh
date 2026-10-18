#!/bin/sh
# 文字モデル（箱 CH）の様子。
#   chlog            要約（事前学習の最後・SFT・NPC 学習の評価と学習ロス EMA）
#   chlog loss [pre|sft|npc]   学習ロスの推移と EMA（省略時は今動いている段階）
#   chlog sft | npc  その段階の評価の推移と返答サンプル
#   chlog chat       事前学習中のチャット評価と返答
#   chlog all        ログ全部
U="https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/pooler_distill/chatsft/audit/boxlog_CH.txt"
L=$(curl -sL "$U?t=$(date +%s)") || { echo "取得失敗"; exit 1; }
sec() { echo "$L" | awk -v a="--- $1 ---" -v b="$2" 'index($0,a)==1{f=1;next} f&&index($0,"--- ")==1&&(b==""||index($0,b)==1){exit} f'; }
pre_log()   { echo "$L" | awk 'index($0,"--- run2/train.log ---")==1{f=1;next} f&&index($0,"--- ")==1{exit} f'; }
steps() {   # $1 = pre|sft|npc
  case "$1" in pre) pre_log | grep -E '^\[step' ;; sft) sec "sft1 steps" "--- " ;; npc) sec "npc1 steps" "--- " ;; esac; }
ema() { awk '{for(i=1;i<=NF;i++){if($i=="train")v=$(i+1); if($i=="min"||$i=="min,")m=$(i-1); if($i=="lr")l=$(i+1)}; gsub(/,/,"",l); gsub(/[^0-9.]/,"",v); if(v=="")next; e=(n++?0.9*e+0.1*v:v); s=$2; gsub(/[\[\]]/,"",s);
  if(mode=="all")printf "step %-12s %5s min  train %.3f  EMA %.3f  lr %s\n",s,m,v,e,l; else if(n%6==0)printf "%5s min  EMA %.3f\n",m,e}
  END{if(n&&mode!="all")printf "最新 %s min  EMA %.3f（生 %.3f）\n",m,e,v}' mode="$1"; }
active() { [ -n "$(steps npc)" ] && echo npc && return; [ -n "$(steps sft)" ] && echo sft && return; echo pre; }
echo "$L" | head -2
case "$1" in
  loss) st=${2:-$(active)}; echo "--- 学習ロス（$st）: 生の値と EMA（α=0.1、約 50 分のならし） ---"; steps "$st" | ema all ;;
  sft|npc) r=$([ "$1" = sft ] && echo sft1 || echo npc1)
        echo "--- $1: 評価（返答部分の bits/文字、低いほど良い） ---"; sec "$r/train.log" "--- " | grep -E '^\[val\]|^\[stop\]|^\[gptq\]'
        echo "--- $1: 直近の返答サンプル ---"; sec "$r/train.log" "--- " | grep -E '^\[reply' | tail -n 6 ;;
  chat) pre_log | grep -E '^\[chat\]|^\[chat-sample\]' | tail -n 12 ;;
  all)  echo "$L" ;;
  *)    echo "--- 事前学習（終了）: 最後の検証 ---"; pre_log | grep -E '^\[val\] step|^\[chat\]|^\[test\]' | tail -n 4
        for st in sft npc; do r=$([ "$st" = sft ] && echo sft1 || echo npc1); v=$(sec "$r/train.log" "--- " | grep -E '^\[val\]|^\[stop\]|NPC_DONE|SFT_DONE')
          [ -z "$v" ] && continue
          echo "--- $st: 評価（返答部分の bits/文字） ---"; echo "$v" | tail -n 5
          echo "--- $st: 学習ロス EMA ---"; steps "$st" | ema | tail -n 4; done
        echo "--- 状態 ---"; echo "$L" | grep -E '^\[npc2[0-9]\]|^\[oom\]|Traceback|CHSFT_DONE|^\[gptq\]' | tail -n 4 ;;
esac
