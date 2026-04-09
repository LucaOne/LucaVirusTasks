# LucaVirus
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_genome_drop_weak.csv \
    --llm_truncation_seq_length 29905 \
    --emb_dir ../../../matrices/DeepAbBindv2_genome/lucavirus/v1.0/3800000 \
    --model_path .. \
    --save_path ../../predicted/LucaVirus/DeepAbBindv2_genome/lucavirus_test_genome_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251004085527 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_nucl_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_nucl/lucavirus/v1.0/3800000\
    --model_path .. \
    --save_path ../../predicted/LucaVirus/DeepAbBindv2_nucl/lucavirus_test_nucl_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251004085817 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_original_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_original/lucavirus/v1.0/3800000 \
    --model_path .. \
    --save_path ../../predicted/LucaVirus/DeepAbBindv2_original/lucavirus_test_original_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251004090016 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2


##############
# LucaVirus-Mask
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_genome_drop_weak.csv \
    --llm_truncation_seq_length 29905 \
    --emb_dir ../../../matrices/DeepAbBindv2_genome/lucavirus-mask/v1.0/3800000 \
    --model_path .. \
    --save_path ../../predicted/LucaVirus-Mask/DeepAbBindv2_genome/lucavirus_test_genome_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20250916221125 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2

# LucaVirus-Mask
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_nucl_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_nucl/lucavirus-mask/v1.0/3800000  \
    --model_path .. \
    --save_path ../../predicted/LucaVirus-Mask/DeepAbBindv2_nucl/lucavirus_test_nucl_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20250916215353 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2

# LucaVirus-Mask
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_original_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_original/lucavirus-mask/v1.0/3800000 \
    --model_path .. \
    --save_path ../../predicted/LucaVirus-Mask/DeepAbBindv2_original/lucavirus_test_original_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20250919155318 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2

#####
# LucaVirus-Gene/Prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_genome_drop_weak.csv \
    --llm_truncation_seq_length 29905 \
    --emb_dir ../../../matrices/DeepAbBindv2_genome/lucavirus-gene/v1.0/3800000#../../../matrices/DeepAbBindv2_genome/lucavirus-prot/v1.0/3800000 \
    --model_path .. \
    --save_path ../../predicted/LucaVirus-Gene-Prot/DeepAbBindv2_genome/lucavirus_test_genome_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251003112812 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2

# LucaVirus-Gene/Prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_nucl_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_nucl/lucavirus-gene/v1.0/3800000#../../../matrices/DeepAbBindv2_nucl/lucavirus-prot/v1.0/3800000\
    --model_path .. \
    --save_path ../../predicted/LucaVirus-Gene-Prot/DeepAbBindv2_nucl/lucavirus_test_nucl_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251001135723 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2

# LucaVirus-Gene/Prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --input_file ../../data/DeepAbBindv2/test_original_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_original/lucavirus-prot/v1.0/3800000 \
    --model_path .. \
    --save_path ../../predicted/LucaVirus-Prot/DeepAbBindv2_original/lucavirus_test_original_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251007093221 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 2




