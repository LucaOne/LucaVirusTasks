cd ../llm/nucleotide_transformer/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_version 2.5b-multi-species \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/genome/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/nucleotide_transformer/multi-species/2.5B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 3


# 255 机器
# DeepAbBindv2_nucl
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_version 2.5b-multi-species \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/nucl/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/nucleotide_transformer/multi-species/2.5B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 3


cd ../esm2/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/genome/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/nucl/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2


# DeepAbBindv2_original
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
   --llm_type esm2 \
   --llm_version 3B \
   --truncation_seq_length 10240 \
   --trunc_type right \
   --seq_type prot \
   --input_file ../../../data/DeepAbBindv2/original/prot.fasta \
   --save_path  ../../../../matrices/DeepAbBindv2_original/esm/esm2/3B \
   --embedding_type matrix \
   --matrix_add_special_token \
   --embedding_complete \
   --embedding_complete_seg_overlap \
   --embedding_fixed_len_a_time 10240 \
   --gpu_id 2



# DeepAbBindv2_original
cd ../lucaone/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone\
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/original/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_original/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2

# DeepAbBindv2_genome
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone\
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/genome/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/genome/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 2


# DeepAbBindv2_nucl
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/nucl/prot.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 17600000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/nucl/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/lucaone/v2.0/17600000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 2