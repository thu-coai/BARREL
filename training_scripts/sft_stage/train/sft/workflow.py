import sys
sys.path.append('..')
from utils.args import ModelArguments, DataArguments, TrainingArguments, GeneratingArguments
from model.load import load_model_and_tokenizer
from transformers import default_data_collator
from .trainer import SFTTrainer
from data.dataset import SafetyDataset, create_dataset

def run_sft(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    generating_args: GeneratingArguments
):
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    
    # load dataset 
    train_dataset, eval_dataset = create_dataset(data_args, tokenizer)
    
    # data collator
    data_collator = default_data_collator

    # Initialize our Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    trainer.train(resume_from_checkpoint=False)