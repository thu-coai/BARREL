set -x

gsm8k_train_path=/data/yangjunxiao/Reasoning_Hallucination/verl/data/processed_data/gsm8k/train.parquet
gsm8k_test_path=/data/yangjunxiao/Reasoning_Hallucination/verl/data/processed_data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

<<<<<<< HEAD

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
=======
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
>>>>>>> 4081d8af1f4c6cbfcf2b27c2556da3345ec9d826
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
<<<<<<< HEAD
    actor_rollout_ref.model.path=/data3/public_checkpoints/huggingface_models/Qwen2-7B-Instruct \
=======
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
>>>>>>> 4081d8af1f4c6cbfcf2b27c2556da3345ec9d826
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/data3/public_checkpoints/huggingface_models/Qwen2-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name='qwen2-7b_function_rm_bsz8k_p4k_r4k_seq_packing' \
<<<<<<< HEAD
    trainer.n_gpus_per_node=4 \
    +trainer.val_before_train=False \
=======
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
>>>>>>> 4081d8af1f4c6cbfcf2b27c2556da3345ec9d826
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
