conda activate webrl
cd /path_to_webrl/scripts

NPROC_PER_NODE=$MLP_GPU
NNODES=$MLP_WORKER_NUM
RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST
MASTER_PORT=$MLP_WORKER_0_PORT

 
export VLLM_USE_MODELSCOPE="False"
export NCCL_IB_TC=160
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=120
export NCCL_TIMEOUT=1200
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none

export NCCL_P2P_DISABLE=0
#export NCCL_P2P_LEVEL=PXB
export NCCL_P2P_LEVEL=NVL

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_MIN_CTAS=4
export NCCL_IB_RETRY_CNT=7

export NCCL_IB_DISABLE=0

echo $NCCL_P2P_LEVEL

export GLOO_SOCKET_IFNAME=eth0 
export CUDA_DEVICE_MAX_CONNECTIONS=1 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    run.py \
    --config_path config/main \
    --config_name webrl  2>&1 | tee /path_to_webrl/multi_node_logs/${MLP_ROLE_INDEX}.log
