

HF_HOME=/root/.cache/hf_home PYTHONPATH=. python3 -m lm_eval \
--model native \
--model_args checkpoint_dir=/mnt/mnhotzc/exp/comp_mem/cm-distneg-mix4_4222-nc16ms256-512-std/updates_1000/,max_seq_length=2048,tokenizer_path=/mnt/msranlphot_intern/liyu/ckpt/Qwen/Qwen3-4B \
--tasks hellaswag \
--output_path results.json \
--batch_size 8 