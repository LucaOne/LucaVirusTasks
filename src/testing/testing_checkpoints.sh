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
    --time_str 20250916215353 \
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

#

###
# LucaOne
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_genome_drop_weak.csv \
    --llm_truncation_seq_length 29905 \
    --emb_dir ../../../matrices/DeepAbBindv2_genome/lucaone/v2.0/17600000 \
    --model_path .. \
    --save_path ../../predicted/LucaOne/DeepAbBindv2_genome/lucaone_test_genome_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_genome \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251005105310 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_nucl_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_nucl/lucaone/v2.0/17600000  \
    --model_path .. \
    --save_path ../../predicted/LucaOne/DeepAbBindv2_nucl/lucaone_test_nucl_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_nucl \
    --dataset_type protein_protein_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251002133947 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_original_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_original/lucaone/v2.0/17600000 \
    --model_path .. \
    --save_path ../../predicted/LucaOne/DeepAbBindv2_original/lucaone_test_original_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251002223348 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

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

# ESM2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_original_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_original/esm/esm2/3B \
    --model_path .. \
    --save_path ../../predicted/ESM2/DeepAbBindv2_original/esm2_test_original_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20251004092455 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3

##############
# ESMC
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python testing_checkpoints.py \
    --seq_type prot \
    --input_file ../../data/DeepAbBindv2/test_original_drop_weak.csv \
    --llm_truncation_seq_length 10240 \
    --emb_dir ../../../matrices/DeepAbBindv2_original/esm/esmc/600M \
    --model_path .. \
    --save_path ../../predicted/ESMC/DeepAbBindv2_original/esmc_test_original_drop_weak_predicted_results.csv \
    --dataset_name DeepAbBindv2_original \
    --dataset_type protein_protein_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucatriple_heter \
    --input_type matrix \
    --input_mode triple \
    --time_str 20250930111256 \
    --ground_truth_idx 9 \
    --threshold 0.5 \
    --print_per_num 1000 \
    --matrix_embedding_exists \
    --gpu_id 3