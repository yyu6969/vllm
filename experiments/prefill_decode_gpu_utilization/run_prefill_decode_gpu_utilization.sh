#!/bin/bash
# experiments/run_prefill_decode_gpu_compute_monitor.sh

# Memory management settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Create logs directory
mkdir -p logs

# Create a timestamp for output
timestamp=$(date +"%Y%m%d_%H%M%S")

# Define where we’ll store the generated charts
output_dir="/work/nvme/bdkz/yyu69/vllm/experiment_results/prefill_decode_gpu_utilization/${timestamp}"
mkdir -p "${output_dir}"

# Define the log file name
log_file="logs/gpu_compute_${timestamp}.log"

echo "Running GPU compute monitor..."
echo "Logs will be saved to: ${log_file}"
echo "Charts will be saved to: ${output_dir}"
echo

# Execute the Python script with the necessary arguments
python -u /work/nvme/bdkz/yyu69/vllm/experiments/prefill_decode_gpu_utilization/prefill_decode_gpu_utilization.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --prompt-files /work/nvme/bdkz/yyu69/vllm/data/prefill_decode/long-prompts-selection-1250-1375.csv /work/nvme/bdkz/yyu69/vllm/data/prefill_decode/long-prompts-selection-2500-2750.csv /work/nvme/bdkz/yyu69/vllm/data/prefill_decode/long-prompts-selection-5000-5500.csv \
    --output-dir "${output_dir}" \
    --max-new-tokens 512 \
    --use-wandb \
    --wandb-project "gpu-compute-monitor" \
    --wandb-run-name "test" \
    2>&1 | tee "${log_file}"

echo
echo "Monitoring complete."
echo "Output charts are located in: ${output_dir}"
echo "Log file is located at: ${log_file}"