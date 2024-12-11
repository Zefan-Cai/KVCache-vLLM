export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python3 test_scripts/test_benchmark.py \
  --model ../models/Llama-3-8B-Instruct-262k \
  --enforce-eager \
  --num-prompts 256 \
  --input-len 7000 \
  --output-len 600 \
  --protected-window-size 32 \
  --compression-rate 64