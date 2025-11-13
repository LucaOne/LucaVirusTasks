prepare environment
conda create -n lucavirus_tasks_dnabert2 python=3.9.13
conda activate lucavirus_tasks_dnabert2
pip install -r requirements_dnabert2.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


to run test:
```
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
```