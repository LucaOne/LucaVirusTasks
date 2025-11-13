todo
# DMS_Bind_Reps_Strain_Nucl
cd ./src/llm/dnabert2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DMS_Bind_Reps_Strain_Nucl/gene/regression/all_genes.fasta \
    --save_path ../../../../matrices/DMS_Bind_Reps_Strain_Nucl/dnabert/dnabert2/117M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0


# DeepAbBindv2_genome
cd ./src/llm/dnabert2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/dnabert/dnabert2/117M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0


cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_genome/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0

# DeepAbBindv2_nucl
cd ./src/llm/dnabert2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_genes.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/dnabert/dnabert2/117M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 0


cd ./src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/DeepAbBindv2_nucl/protein_protein_gene/binary_class/all_proteins.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/esm/esm2/3B \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 10240 \
    --gpu_id 3

