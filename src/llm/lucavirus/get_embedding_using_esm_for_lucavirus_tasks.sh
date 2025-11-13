# 129机器
# VirusEC4
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
  --input_file ../../../dataset/VirusEC4/protein/multi_label/all_proteins.fasta \
  --save_path ../../../../matrices/VirusEC4/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 0

# 255机器
# VirusEC4
cd ./src/llm/esmc
python predict_embedding.py \
    --llm_type esmc \
    --llm_version 600M \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VirusEC4/protein/multi_label/all_proteins.fasta \
    --save_path  ../../../../matrices/VirusEC4/esm/esmc/600M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# 142机器
# ViralCapsid
cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/ViralCapsid/protein/binary_class/all_proteins.fasta \
   --save_path ../../../../matrices/ViralCapsid/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 4098 \
   --gpu_id 1

# 142机器
# ViralCapsid
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esmc \
   --llm_version 600M \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/ViralCapsid/protein/binary_class/all_proteins.fasta \
   --save_path  ../../../../matrices/ViralCapsid/esm/esmc/600M \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 4098 \
   --gpu_id 1

# 64机器
# RdRP
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/RdRP/protein/binary_class/all_proteins.fasta \
   --save_path  ../../../../matrices/RdRP/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 4098 \
   --gpu_id 0

# 142机器
# RdRP
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esmc \
   --llm_version 600M \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/RdRP/protein/binary_class/all_proteins.fasta \
   --save_path  ../../../../matrices/RdRP/esm/esmc/600M \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 4098 \
   --gpu_id 1


# 129机器
# DMS_Bind_Reps_Strain
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/DMS_Bind_Reps_Strain/protein/regression/all_proteins.fasta \
   --save_path  ../../../../matrices/DMS_Bind_Reps_Strain/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 1

# 142机器
# DMS_Bind_Reps_Strain
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esmc \
   --llm_version 600M \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/DMS_Bind_Reps_Strain/protein/regression/all_proteins.fasta \
   --save_path  ../../../../matrices/DMS_Bind_Reps_Strain/esm/esmc/600M \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 1


# 129机器
# DeepAbBindv2_original
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/DeepAbBindv2_original/protein_protein_protein/binary_class/all_proteins.fasta \
   --save_path  ../../../../matrices/DeepAbBindv2_original/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 1

# 142机器
# DeepAbBindv2_original
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esmc \
   --llm_version 600M \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/DeepAbBindv2_original/protein_protein_protein/binary_class/all_proteins.fasta \
   --save_path  ../../../../matrices/DeepAbBindv2_original/esm/esmc/600M \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 1

# 122机器
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/DMS_EVA/protein/regression/all_proteins.fasta \
   --save_path  ../../../../matrices/DeepAbBindv2_original/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 1

# 122机器
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esmc \
   --llm_version 600M \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../dataset/DMS_EVA/protein/regression/all_proteins.fasta \
   --save_path  ../../../../matrices/DMS_EVA/esm/esmc/600M \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 1