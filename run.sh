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


python /work/nvme/bdkz/yyu69/vllm/experiments/utiles/token_counter.py --model Qwen/Qwen2.5-14B-Instruct --mode csv --input /work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/data_1_500.csv --prompt-field text

python /work/nvme/bdkz/yyu69/vllm/experiments/utiles/token_counter.py --model Qwen/Qwen2.5-14B-Instruct --mode csv --input /work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/data_1_8000.csv --prompt-field text

python /work/nvme/bdkz/yyu69/vllm/experiments/utiles/token_counter.py --model Qwen/Qwen2.5-14B-Instruct --mode csv --input /work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/data_2_4000.csv --prompt-field text

python /work/nvme/bdkz/yyu69/vllm/experiments/utiles/token_counter.py --model Qwen/Qwen2.5-14B-Instruct --mode csv --input /work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/data_2_1000.csv --prompt-field text

python /work/nvme/bdkz/yyu69/vllm/experiments/utiles/token_counter.py --model Qwen/Qwen2.5-14B-Instruct --mode csv --input /work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/dataset_2/data_8000.csv --prompt-field prompt

python /work/nvme/bdkz/yyu69/vllm/experiments/utiles/token_counter.py --model Qwen/Qwen2.5-14B-Instruct --mode csv --input /work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/dataset_2/long-prompts-selection.csv --prompt-field prompt




results_json_path = "/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_vs_e2e_time_experiments/chunk_size_vs_e2e_time_experiment_20250427_032248/Qwen2.5-14B-Instruct/chunk_size_results_Qwen2.5-14B-Instruct.json"
model_dir = "/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_vs_e2e_time_experiments/chunk_size_vs_e2e_time_experiment_20250427_032248/Qwen2.5-14B-Instruct"
model_short_name = "Qwen2.5-14B-Instruct"

plot_e2e_time_chart_from_json(results_json_path, model_dir, model_short_name)
plot_ttft_time_chart_from_json(results_json_path, model_dir, model_short_name)
plot_tbt_time_chart_from_json(results_json_path, model_dir, model_short_name)