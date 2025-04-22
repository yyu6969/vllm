conda activate vllm
module load cuda/12.4
cd /work/nvme/bdkz/yyu69
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false