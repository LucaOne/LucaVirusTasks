prepare environment
conda create -n lucavirus_tasks_esmc python=3.10.16
conda activate lucavirus_tasks_esmc
pip install -r requirements_esmc.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


to run test:
```
cd src/llm/esmc/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esmc \
    --llm_version 600M \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/DeepAbBindv2/DeepAbBindv2_all_single_prot.fasta \
    --save_path  /mnt2/sanyuan.hy/matrices/DeepAbBindv2/esm/esmc/600M \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 4096 \
    --gpu_id 3
```