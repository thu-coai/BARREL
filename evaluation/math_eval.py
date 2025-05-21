import json
import os
from collections import defaultdict
from utils import answers_match, extract_model_answer, check_unknown
from math_utils import check_is_correct, extract_answer

# ======================= Configurations ======================= #
# Change these values if you need to run on different settings
name = "math500_test"
model = "DeepSeek-R1-Distill-Llama-8B"

# Construct input and output paths
data_path = f"../gen_code/gen_results/{model}/{name}.json"
output_path = f"./eval_results/{model}/{name}.json"

# ======================= Utility Functions ======================= #

def ensure_dir(path):
    """Ensure the directory for the given file path exists."""
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def process_item(item):
    """
    Process a single data item and evaluate correctness and unknownness.

    Parameters:
    - item: dictionary containing response, final_answer, known, etc.

    Returns:
    - result dictionary with evaluation flags and extracted answer.
    """
    prompt = item.get("prompt", "")
    response = item.get("response", "")
    reasoning = item.get("reasoning", "")
    final_answer = item.get("final_answer", "")
    known = item.get("known", None)
    source = item.get("source", "")  # Defaults to empty string if missing
    # Extract the answer from model prediction
    extracted_answer = extract_answer(response)

    if isinstance(final_answer, str):
        final_answer = [final_answer]

    # Check if the model answer matches any of the gold final answers
    is_correct = any(check_is_correct(extracted_answer, ans) for ans in final_answer)

    result = {
        "prompt": prompt,
        "response": response,
        "reasoning": reasoning,
        "known": known,
        "final_answer": final_answer,
        "exacted_answer": extracted_answer,
        "is_correct": is_correct,
        "is_unknown": check_unknown(response),
        "source": source
    }

    return result


def compute_metrics(results):
    """
    Compute key evaluation metrics:
    - Accuracy (Acc.) = N_c / N
    - Truthfulness (Truth.) = (N_c + N_r) / N
    - Relevance (Rel.) = ans. * Truth. + (1 - ans.) * Acc.
      where ans. = 1 - N_r / N

    Also computes:
    - Missed unknown rate
    - Wrong rate

    Returns:
    - Dictionary of all metrics.
    """
    N = len(results)  # Total number of samples
    N_c = 0           # Number of correct answers
    N_w = 0           # Number of incorrect answers (known)
    N_r = 0           # Anti-boundary errors: model falsely claims knowledge
    N_u = 0           # Properly flagged unknowns
    miss_unknown = 0  # Known cases wrongly labeled as unknown

    for r in results:
        known = r.get("known")
        is_correct = r.get("is_correct")
        is_unknown = r.get("is_unknown")

        if known is True:
            if is_unknown:
                miss_unknown += 1
            
        if is_correct:
            N_c += 1
        elif is_unknown:
            N_r += 1 
        else:
            N_w += 1

    # Compute core metrics based on mathematical definitions
    Acc = N_c / N if N else 0
    Truth = (N_c + N_r) / N if N else 0
    ans = 1 - (N_r / N) if N else 0
    Rel = ans * Truth + (1 - ans) * Acc if N else 0

    # Extra diagnostics
    miss_unknown_rate = miss_unknown / N if N else 0
    wrong_rate = N_w / N if N else 0

    return {
        "accuracy": Acc,
        "truthfulness": Truth,
        "reliable": Rel,
        "miss_unknown_rate": miss_unknown_rate,
        "wrong_rate": wrong_rate
    }


# ======================= Main Process ======================= #

def main():
    """Main workflow for processing input data and evaluating metrics."""
    ensure_dir(output_path)

    # Load dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each item in the dataset
    processed_results = [process_item(item) for item in data]

    # Group results by their source type
    source_results = defaultdict(list)
    for r in processed_results:
        source_results[r["source"]].append(r)

    # Compute overall metrics
    overall_metrics = compute_metrics(processed_results)

    # Compute metrics for each data source
    source_metrics = {}
    for source, group in source_results.items():
        source_metrics[source] = compute_metrics(group)

    # Prepare output dictionary
    output_data = {
        "overall_metrics": overall_metrics,
        "source_metrics": source_metrics,
        "results": processed_results
    }

    # Save output as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Print results to console
    print("Overall Metrics:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value * 100:.2f}%")

    print("\nMetrics by Source:")
    for source, metrics in source_metrics.items():
        print(f"\nSource: {source}")
        for key, value in metrics.items():
            print(f"  {key}: {value * 100:.2f}%")


# ======================= Entry Point ======================= #

if __name__ == "__main__":
    main()