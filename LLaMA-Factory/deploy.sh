source /workspace/qzh/miniconda3/etc/profile.d/conda.sh
conda activate code
vllm serve /checkpoints/qzh/llama3.1-70B \
--tensor-parallel-size 8 \
--trust-remote-code