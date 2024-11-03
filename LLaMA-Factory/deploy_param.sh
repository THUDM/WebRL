source /workspace/qzh/miniconda3/etc/profile.d/conda.sh
conda activate code

path=$1

echo ${path}

vllm serve ${path} \
--tensor-parallel-size 8 \
--trust-remote-code