import json
import argparse
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_model import VLLMModel
from vllm import LLM, SamplingParams
from loguru import logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate text using a pretrained model.")
    parser.add_argument('--base_model', type=str, required=True, help='Path to the base model')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file containing prompts')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output file')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of prompts to process')
    parser.add_argument('--regen', type=int, default=1, help='Regeneration flag')
    parser.add_argument('--temperature', type=float, default=0.6, help='Regeneration flag')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of new tokens to generate')
    return parser.parse_args()

def load_prompts(input_file, limit):
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts[:limit] if limit > 0 else prompts

def save_results(output_file, results):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def load_existing_outputs(output_file):
    # Load existing outputs if output_file exists
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return []
    
def main():
    args = parse_arguments()

    # Load model and tokenizer
    logger.info(f"Loading model from {args.base_model}")
    model = LLM(model=args.base_model, tokenizer=args.tokenizer_path,  gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side='left', trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    generation_config = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    model = VLLMModel(model=model, tokenizer=tokenizer, model_name=args.base_model, generation_config=generation_config)

    # Load input prompts
    logger.info(f"Loading input prompts from {args.input_file}")
    prompts = load_prompts(args.input_file, args.limit)[:]

    if args.regen == 0:
        logger.info(f"Checking existing outputs at {args.output_file}")
        existing_results = load_existing_outputs(args.output_file)
        existing_ids = set(item["id"] for item in existing_results if "id" in item)
        new_prompts = [p for p in prompts if p["id"] not in existing_ids]
    else:
        existing_ids = []
        existing_results = []
        new_prompts = prompts
        
    logger.info(f"{len(existing_ids)} prompts already processed. {len(new_prompts)} prompts left to generate.")
    if not new_prompts:
        # logger.info(f"Saving results to {args.output_file}")
        # save_results(args.output_file, existing_results)
        # logger.info("No new prompts to generate. Exiting.")
        return
    
    # Generate responses
    logger.info("Generating responses...")
    responses = model.batch_chat([item["prompt"] for item in new_prompts], generation_config=generation_config)

    new_results = []
    for (prompt, response) in zip(new_prompts, responses):
        if r"</think>" in response:
            reasoning, response_content = response.split(r"</think>", 1)
            new_results.append(
                {
                    "id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "final_answer": prompt["final_answer"],
                    "response": response_content.strip(),
                    "reasoning": reasoning.strip()
                }
            )
        else:
            try:
                new_results.append(
                    {
                        "id": prompt["id"],
                        "prompt": prompt["prompt"],
                        "final_answer": prompt["final_answer"],
                        "response": response
                    }
                )
            except:
                new_results.append(
                    {
                        "id": prompt["id"],
                        "prompt": prompt["prompt"],
                        "response": response
                    }
                )
        try:
            new_results[-1]["known"] = prompt["known"]
        except:
            pass
        try:
            new_results[-1]["source"] = prompt["source"]
        except:
            pass
        
    
    all_results = existing_results + new_results
    # Save results
    logger.info(f"Saving results to {args.output_file}")
    save_results(args.output_file, all_results)
    logger.info("Generation completed successfully.")

if __name__ == "__main__":
    main()