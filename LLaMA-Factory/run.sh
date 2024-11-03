yaml=$1

conda activate <llama_factory>
cd /path_to_llama_factory
echo ${yaml}
llamafactory-cli train ${yaml}