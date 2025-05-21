# usage: replace the following variable with your local path and dataset

model_name=DeepSeek-R1-Distill-Llama-8B # Llama-3.1-8B-Instruct
base_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
tokenizer_path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B

input_file=../data/test_data/test_3k/test_3k_inst.json
output_file=./gen_results/test_data/${model_name}/test_3k_inst.json

echo ${input_file}
echo ${output_file}
CUDA_VISIBLE_DEVICES=4 python generate.py --base_model ${base_model} --tokenizer_path ${tokenizer_path} --input_file ${input_file} --output_file ${output_file} --limit 0 --regen 1 --max_new_tokens 4096 --temperature 0.6