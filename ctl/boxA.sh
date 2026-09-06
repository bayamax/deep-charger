# box A: pin a transformers that supports torch 2.4 and the newer tokenizer.json, then launch gen
pip install -q "transformers==4.51.3" "tokenizers==0.21.1" "peft==0.15.2" 2>&1 | tail -1
python3 -c "import torch,transformers,tokenizers,peft; print('versions torch',torch.__version__, transformers.__version__, tokenizers.__version__, peft.__version__)"
python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; t=AutoTokenizer.from_pretrained('/root/work/bf16_mus_pure'); print('tokenizer ok', len(t))"
bash /root/do_gen.sh
