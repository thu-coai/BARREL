from typing import Any, Dict, Optional
from train.sft.workflow import run_sft
from utils.argparser import get_train_args

def run(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, training_args, generating_args = get_train_args(args)

    if training_args.method == "sft":
        run_sft(model_args, data_args, training_args, generating_args)
    else:
        raise ValueError(f"Unknown Training Method: {training_args.method}.")

def main():
    """Entry point of the script."""
    run()

if __name__ == "__main__":
    main()