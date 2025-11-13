# 122机器
# HIV_Entry_Latent_Fold_00
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_Entry_Latent_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_Entry_Latent_Fold_00/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 5

# HIV_Entry_Observed_Fold_00
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_Entry_Observed_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_Entry_Observed_Fold_00/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 5

# HIV_immune_escape_10_1074_Fold_00
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_immune_escape_10_1074_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_immune_escape_10_1074_Fold_00/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 5

# HIV_immune_escape_3BNC117_Fold_00
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HIV_immune_escape_3BNC117_Fold_00/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HIV_immune_escape_3BNC117_Fold_00/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 5

# HIV_immune_escape_sera_mean_Fold_00
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
  --llm_type esm2 \
  --llm_version 3B \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/HIV_immune_escape_sera_mean_Fold_00/protein/regression/all_seqs.fasta \
  --save_path ../../../../matrices/HIV_immune_escape_sera_mean_Fold_00/esm/esm2/3B \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 10242 \
  --gpu_id 5

# HK19_immune_escape_per_serum
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/HK19_immune_escape_per_serum/protein/regression/all_seqs.fasta \
    --save_path ../../../../matrices/HK19_immune_escape_per_serum/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10242 \
    --gpu_id 5

# PE09_immune_escape_per_serum
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
  --llm_type esm2 \
  --llm_version 3B \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/PE09_immune_escape_per_serum/protein/regression/all_seqs.fasta \
  --save_path ../../../../matrices/PE09_immune_escape_per_serum/esm/esm2/3B \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 10242 \
  --gpu_id 5

# Phage_PVP
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python predict_embedding.py \
  --llm_type esm2 \
  --llm_version 3B \
  --truncation_seq_length 10240 \
  --trunc_type right \
  --seq_type prot \
  --input_file ../../../dataset/Phage_PVP/protein/multi_class/all_seqs.fasta \
  --save_path ../../../../matrices/Phage_PVP/esm/esm2/3B \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --embedding_fixed_len_a_time 4096 \
  --gpu_id 1