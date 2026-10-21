#!/bin/bash
# Box VA (3060, vLLM image): villager data for the char model (github.com/bayamax/charlm-agent): a held-out villager
# bench (villagerbench) and villager talks (villager) from Qwen3.5-4B, written chunk by chunk and mirrored to the hub
# (baya1116/hypernet-sp-distill, archive/charlm/villager/) as they come; the generation stops after 32 minutes
BOXVA_SERIAL=2
if [ -f /root/.boxva_serial ] && [ "$(cat /root/.boxva_serial)" -gt "$BOXVA_SERIAL" ] 2>/dev/null; then echo "BOXVA_STALE $BOXVA_SERIAL"; exit 0; fi
echo $BOXVA_SERIAL > /root/.boxva_serial
if [ ! -e /root/.va_pre ]; then touch /root/.va_pre
  echo "[va] preflight GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader)"
  echo "[va] preflight hub: $(curl -sSL -o /dev/null -m 90 -w '%{size_download} bytes in %{time_total}s' https://huggingface.co/baya1116/charlm-agent/resolve/main/agent6/model_final.pt -r 0-104857599 2>&1)"
fi
[ -d /venv/main/bin ] && export PATH=/venv/main/bin:$PATH
for f in $(grep -l -i vllm /etc/supervisor/conf.d/* 2>/dev/null); do mv $f $f.off; done
supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1; supervisorctl stop all > /dev/null 2>&1; pkill -f "vllm serve" 2>/dev/null
[ -s /root/.hf_token ] || { [ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token; }
echo "[va] HF token: $([ -s /root/.hf_token ] && echo present || echo MISSING)"
RAW="https://raw.githubusercontent.com/bayamax/charlm-agent/main"
mkdir -p /root/va
for f in npc_gen.py; do
  for try in 1 2 3; do curl -sS -o /root/va/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/va/$f && break; sleep 5; done
done
if [ ! -f /root/.bootstrapped ]; then
  python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1
  python3 -c "import vllm, torch; print('[va] vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tail -1
  touch /root/.bootstrapped; echo "BOOTSTRAP_DONE $(date -u)"
fi
cat > /root/status.sh <<'ST'
grep -E "^\[(va|gen|hub)\]|Error|Traceback" /root/va/va.log 2>/dev/null | tail -n 6 | cut -c1-300
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ST
chmod +x /root/status.sh

if [ ! -e /root/.vajob ]; then touch /root/.vajob
  cat > /root/vajob.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/va
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
GE="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.9 --max-seqs 16 --eager 1"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/villager/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
gen() {   # mode n seed out chunk: the normal settings, then the light ones
  gpu_clear; python3 -u npc_gen.py $1 --n $2 --seed $3 --out $4 --chunk $5 $G
  [ -s $4 ] || { echo "[va] $4: again with the light settings"; gpu_clear; python3 -u npc_gen.py $1 --n $2 --seed $3 --out $4 --chunk $5 $GE; }
}
gen villagerbench 70 1 vbench.jsonl 70; put vbench.jsonl vbench.jsonl; put vbench.jsonl.raw vbench.jsonl.raw
( while sleep 240; do put villager.jsonl villager.jsonl > /dev/null; done ) & MIR=$!
export -f gen gpu_clear; export G GE
timeout 32m bash -c 'gen villager 2000 0 villager.jsonl 200'
kill $MIR; put villager.jsonl villager.jsonl; put villager.jsonl.raw villager.jsonl.raw
echo "[va] $(wc -l < villager.jsonl) villager conversations, $(wc -l < vbench.jsonl) bench villagers"
date -u > DONE; put DONE DONE; echo "[va] VAJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vajob.sh >> /root/va/va.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "VAJOB_LAUNCHED $(date -u)"
fi
# serial 2, the user: the model is for many games (a cozy life-sim, a friend-collection sim, a dark fantasy): the cozy run
# stopped (what it made kept as villager_cozy.jsonl), then a bench and talks for the friend-sim and the dark fantasy
if [ ! -e /root/.va_s2 ]; then touch /root/.va_s2
  pkill -f "vajo[b].sh"; pkill -f "npc_ge[n].py"; sleep 5; fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3
  cd /root/va && mv villager.jsonl villager_cozy.jsonl && mv villager.jsonl.raw villager_cozy.jsonl.raw; cd /
  for f in npc_gen.py; do curl -sS -o /root/va/$f "$RAW/$f?nocache=$(date +%s)" && python3 -m py_compile /root/va/$f; done
  cat > /root/vajob2.sh <<'JOB'
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token 2>/dev/null); cd /root/va
G="--model QuantTrio/Qwen3.5-4B-AWQ --fallback cyankiwi/Qwen3.5-4B-AWQ-4bit --max-len 4096 --gpu-mem 0.88 --max-seqs 48"
gpu_clear() { fuser -k -9 /dev/nvidia* > /dev/null 2>&1; sleep 3; }
put() { python3 -c "import sys; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='archive/charlm/villager/'+sys.argv[2], repo_id='baya1116/hypernet-sp-distill')" "$1" "$2" > /dev/null 2>&1 && echo "[hub] $2" || echo "[hub] $2 FAILED"; }
put villager_cozy.jsonl villager_cozy.jsonl; put villager_cozy.jsonl.raw villager_cozy.jsonl.raw
gpu_clear; python3 -u npc_gen.py villagerbench --genres friends,dark --n 60 --seed 2 --out vbench_gen.jsonl --chunk 60 $G
put vbench_gen.jsonl vbench_gen.jsonl
( while sleep 240; do put villager_gen.jsonl villager_gen.jsonl > /dev/null; done ) & MIR=$!
gpu_clear; timeout 22m python3 -u npc_gen.py villager --genres friends,dark --n 1600 --seed 3 --out villager_gen.jsonl --chunk 200 $G
kill $MIR; put villager_gen.jsonl villager_gen.jsonl; put villager_gen.jsonl.raw villager_gen.jsonl.raw
echo "[va] cozy $(wc -l < villager_cozy.jsonl), other games $(wc -l < villager_gen.jsonl) conversations; bench $(wc -l < vbench.jsonl) + $(wc -l < vbench_gen.jsonl)"
date -u > DONE; put DONE DONE; echo "[va] VAJOB_DONE $(date -u)"
JOB
  setsid nohup bash -c 'bash /root/vajob2.sh >> /root/va/va.log 2>&1' > /dev/null 2>&1 < /dev/null &
  echo "[va] s2: cozy run stopped and kept; friend-sim and dark fantasy started $(date -u)"
fi
echo "BOXVA_OK serial $BOXVA_SERIAL $(date -u)"
# CTL-END
