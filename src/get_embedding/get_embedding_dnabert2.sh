# DeepAbBindv2_genome
cd ../llm/dnabert2/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/genome/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_genome/dnabert/dnabert2/117M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 2


# DeepAbBindv2_nucl
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/DeepAbBindv2/nucl/gene.fasta \
    --save_path ../../../../matrices/DeepAbBindv2_nucl/dnabert/dnabert2/117M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 2
