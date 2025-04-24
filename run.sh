conda activate vllm
module load cuda/12.4
cd /work/nvme/bdkz/yyu69
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false


python /work/nvme/bdkz/yyu69/vllm/experiments/prefill_decode_gpu_utilization/prefill_decode_gpu_utilization.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --prompt-files prompts_short.csv prompts_medium.csv prompts_long.csv \
  --csv-column "question" \
  --output-dir "./gpu_utilization_results" \
  --max-new-tokens 256 \
  --use-wandb