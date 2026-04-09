
##############
# ESM2 + DNABert2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_genome_drop_weak.csv \
    --llm_truncation_seq_length 29905 \
    --emb_dir ../../../matrices/DeepAbBindv2_genome/esm/esm2/3B#../../../matrices/DeepAbBindv2_genome/dnabert/dnabert2/117M \
    --model_path .. \
    --save_path ../../predicted/DNABert2-ESM2/DeepAbBindv2_genome/dnabert2_esm2_test_genome_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251007101340 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

# ESM2 + NT
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_genome_drop_weak.csv \
    --llm_truncation_seq_length 29905 \
    --emb_dir ../../../matrices/DeepAbBindv2_genome/esm/esm2/3B#../../../matrices/DeepAbBindv2_genome/nucleotide_transformer/multi-species/2.5B \
    --model_path .. \
    --save_path ../../predicted/NT-ESM2/DeepAbBindv2_genome/nt_esm2_test_genome_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251031112323 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

# ESM2 + DNABert2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_nucl_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_nucl/esm/esm2/3B#../../../matrices/DeepAbBindv2_nucl/dnabert/dnabert2/117M \
    --model_path .. \
    --save_path ../../predicted/DNABert2-ESM2/DeepAbBindv2_nucl/dnabert2_esm2_test_nucl_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251007101531 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

# ESM2 + NT
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_nucl_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_nucl/esm/esm2/3B#../../../matrices/DeepAbBindv2_nucl/nucleotide_transformer/multi-species/2.5B \
    --model_path .. \
    --save_path ../../predicted/NT-ESM2/DeepAbBindv2_nucl/nt_esm2_test_nucl_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251031113451 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3
