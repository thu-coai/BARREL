# BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs

![method](<imgs/final_method.png>)

This is the codebase for our paper [BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs](https://arxiv.org/abs/2505.13529).

We identify two pathological reasoning patterns characterized by overthinking that contribute to the overconfident and incorrect answers: last-minute guessing and second-thought spiraling. To address these issues, we propose BARREL—a novel framework that promotes concise and boundary-aware factual reasoning. Our experiments show that BARREL-training increases the reliability of DeepSeek-R1-Distill-Llama-8B from 39.33\% to 61.48\%, while still achieving accuracy comparable to models finetuned on reasoning data generated by R1. These results demonstrate that our pilot study is inspiring to build more reliable and factual System 2 LRMs. Please refer to our [paper](https://arxiv.org/abs/2505.13529) for more details.

More detailed code and data for data construction and ablation analysis will be available soon. Please feel free to contact us if you have any questions regarding our code and other implementation details.

## Quick Start

### Setup

```shell
git clone git@github.com:thu-coai/BARREL.git
cd BARREL
pip install -r requirements.txt
```

### Evaluation

We provide a script to support batch generation of model responses using the vLLM toolkit. Before running the script, make sure to update the `model/tokenizer` path as well as the `input/output` paths accordingly.

To run the generation:

```bash
cd gen_code
bash generate.sh
```

We also provide quick evaluation scripts to compute factual scores on relevant datasets and accuracy scores on math datasets. Before running the evaluations, remember to update the `name`, `model`, `data_path`, and `output_path` fields in both `factual_eval.py` and `math_eval.py` to reflect your own generation results.

To run the evaluation:

```bash
cd evaluation
python factual_eval.py
python math_eval.py
```

### Training
We provide training scripts for both the SFT stage and the GRPO stage. Before running the scripts, make sure to adjust the configurations as needed (e.g., model path, data path, output directories, hyperparameters, etc.).

#### SFT Stage

To run SFT training:
```bash
cd ./training_scripts/sft_stage
bash run_sft.sh
```

#### GRPO Stage

To run GRPO training:
```bash
cd ./training_scripts/grpo_stage
bash run_grpo.sh
```
Remember to change the default paths to the SFT-trained model and tokenizer.

![result](<imgs/res.png>)

Please feel free to contact us if you have any questions regarding our code and other implementation details.

## Citation
```
@misc{yang2025barrelboundaryawarereasoningfactual,
      title={BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs}, 
      author={Junxiao Yang and Jinzhe Tu and Haoran Liu and Xiaoce Wang and Chujie Zheng and Zhexin Zhang and Shiyao Cui and Caishun Chen and Tiantian He and Hongning Wang and Yew-Soon Ong and Minlie Huang},
      year={2025},
      eprint={2505.13529},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.13529}, 
}
```