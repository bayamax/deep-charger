# box B: install packages the harness imports lazily, rerun smoke
pip install -q datasets scipy scikit-learn sentencepiece protobuf certifi 2>&1 | tail -1
python3 -c "import datasets, numpy, bitsandbytes; print('pkgs ok', datasets.__version__)"
bash /root/do_sft_pool.sh smoke
