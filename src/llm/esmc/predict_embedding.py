#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/12/11 11:03
@project: LucaVirusTasks
@file: predict_embedding
@desc: embedding inference for ESMC
"""
import os
import sys
import torch
import numpy as np
import random, argparse
from timeit import default_timer as timer
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader, csv_reader, tsv_reader
    from utils import calc_emb_filename_by_seq_id
except ImportError:
    from src.file_operator import fasta_reader, csv_reader, tsv_reader
    from src.utils import calc_emb_filename_by_seq_id


def enable_cpu_offloading(model):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:%d" % (7000 + random.randint(0, 1000)), world_size=1, rank=0
    )
    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


esmc_global_model, esmc_global_version = None, None


def complete_embedding_matrix(
        seq_id,
        seq_type,
        seq,
        truncation_seq_length,
        init_emb,
        model_args,
        embedding_type,
        use_cpu=False,
        token=None
):
    """
    :param seq_id:
    :param seq_type:
    :param seq:
    :param truncation_seq_length:
    :param init_emb:
    :param model_args:
    :param embedding_type:
    :param use_cpu:
    :param token:
    """
    if init_emb is not None and model_args.embedding_complete and ("representations" in embedding_type or "matrix" in embedding_type):
        torch.cuda.empty_cache()
        ori_seq_len = len(seq)
        # 每次能处理这么长度
        # print("init_emb:", init_emb.shape)
        cur_segment_len = init_emb.shape[0]
        if model_args.matrix_add_special_token:
            first_emb = init_emb[1:cur_segment_len - 1]
        else:
            first_emb = init_emb
        if model_args.matrix_add_special_token:
            cur_segment_len = cur_segment_len - 2
        # print("cur_segment_len: %d" % cur_segment_len)
        init_cur_segment_len = cur_segment_len
        segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
        if segment_num <= 1:
            return init_emb
        append_emb = None
        if model_args.embedding_complete_seg_overlap:
            sliding_window = init_cur_segment_len // 2
            print("Embedding Complete Seg Overlap: %r, ori seq len: %d, segment len: %d, init sliding window: %d" % (
                model_args.embedding_complete_seg_overlap,
                ori_seq_len, init_cur_segment_len, sliding_window
            ))
            while True:
                print("updated window: %d" % sliding_window)
                try:
                    # 第一个已经处理，滑动窗口
                    if model_args.trunc_type == "right":
                        last_end = init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(init_cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                            seg_idx += 1
                            last_end = min(pos_idx + sliding_window, ori_seq_len)
                            seg_seq = seq[pos_idx - sliding_window:last_end]
                            print("segment idx: %d, seg seq len: %d, ori_seq_len: %d" % (seg_idx, len(seg_seq), ori_seq_len))
                            seg_emb, seg_processed_seq_len = predict_embedding(
                                sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type=model_args.trunc_type,
                                embedding_type=embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                device=model_args.device if not use_cpu else torch.device("cpu"),
                                version=model_args.llm_version,
                                matrix_add_special_token=False,
                                token=token
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[sliding_window:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[sliding_window:]), axis=0)
                        if last_end < ori_seq_len:
                            seg_idx += 1
                            remain = ori_seq_len - last_end
                            seg_seq = seq[ori_seq_len - 2 * sliding_window:ori_seq_len]
                            print("last segment idx: %d, seg seq len: %d, ori_seq_len: %d" % (seg_idx, len(seg_seq), ori_seq_len))
                            seg_emb, seg_processed_seq_len = predict_embedding(
                                sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type=model_args.trunc_type,
                                embedding_type=embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                device=model_args.device if not use_cpu else torch.device("cpu"),
                                version=model_args.llm_version,
                                matrix_add_special_token=False,
                                token=token
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[-remain:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[-remain:]), axis=0)
                    else:
                        last_start = -init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(-init_cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                            seg_idx += 1
                            last_start = max(pos_idx - sliding_window, -ori_seq_len)
                            seg_seq = seq[last_start: pos_idx + sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding(
                                sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type=model_args.trunc_type,
                                embedding_type=embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                device=model_args.device if not use_cpu else torch.device("cpu"),
                                version=model_args.llm_version,
                                matrix_add_special_token=False,
                                token=token
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:sliding_window]
                            else:
                                append_emb = np.concatenate((seg_emb[:sliding_window], append_emb), axis=0)
                        if last_start > -ori_seq_len:
                            seg_idx += 1
                            remain = last_start + ori_seq_len
                            seg_seq = seq[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding(
                                sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type=model_args.trunc_type,
                                embedding_type=embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                device=model_args.device if not use_cpu else torch.device("cpu"),
                                version=model_args.llm_version,
                                matrix_add_special_token=False,
                                token=token
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:remain]
                            else:
                                append_emb = np.concatenate((seg_emb[:remain], append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change sliding window: %d -> %d" % (sliding_window, int(sliding_window * 0.95)))
                sliding_window = int(sliding_window * 0.95)
        else:
            while True:
                print("ori seq len: %d, segment len: %d" % (ori_seq_len, cur_segment_len))
                try:
                    # 第一个已经处理，最后一个单独处理（需要向左/向右扩充至cur_segment_len长度）
                    if model_args.trunc_type == "right":
                        begin_seq_idx = 0
                    else:
                        begin_seq_idx = ori_seq_len - (segment_num - 1) * cur_segment_len
                    for seg_idx in range(1, segment_num - 1):
                        seg_seq = seq[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                        # print("segment idx: %d, seg_seq(%d): %s" % (seg_idx, len(seg_seq), seg_seq))
                        print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                        seg_emb, seg_processed_seq_len = predict_embedding(
                            sample=[seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                            trunc_type=model_args.trunc_type,
                            embedding_type=embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device if not use_cpu else torch.device("cpu"),
                            version=model_args.llm_version,
                            matrix_add_special_token=False,
                            token=token
                        )

                        if append_emb is None:
                            append_emb = seg_emb
                        else:
                            '''
                            if model_args.trunc_type == "right":
                                append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                            else:
                                append_emb = np.concatenate((seg_emb, append_emb), axis=0)
                            '''
                            append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                    if model_args.trunc_type == "right":
                        # 处理最后一个
                        last_seg_seq = seq[-cur_segment_len:]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("last seg seq: %s" % last_seg_seq)
                        print("last seg seq len: %d, really len: %d" % (len(last_seg_seq), really_len))
                        last_seg_emb, last_seg_processed_seq_len = predict_embedding(
                            sample=[seq_id + "_seg_%d" % (segment_num - 1), seq_type, last_seg_seq],
                            trunc_type=model_args.trunc_type,
                            embedding_type=embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device if not use_cpu else torch.device("cpu"),
                            version=model_args.llm_version,
                            matrix_add_special_token=False,
                            token=token
                        )
                        last_seg_emb = last_seg_emb[-really_len:, :]
                        append_emb = np.concatenate((append_emb, last_seg_emb), axis=0)
                    else:
                        # 处理第一个
                        first_seg_seq = seq[:cur_segment_len]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("first seg seq: %s" % first_seg_seq)
                        print("first seg seq len: %d, really len: %d" % (len(first_seg_seq), really_len))
                        first_seg_emb, first_seg_processed_seq_len = predict_embedding(
                            sample=[seq_id + "_seg_0", seq_type, first_seg_seq],
                            trunc_type=model_args.trunc_type,
                            embedding_type=embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device if not use_cpu else torch.device("cpu"),
                            version=model_args.llm_version,
                            matrix_add_special_token=False,
                            token=token
                        )
                        first_seg_emb = first_seg_emb[:really_len, :]
                        append_emb = np.concatenate((first_seg_emb, append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change segment len: %d -> %d, change seg num: %d -> %d" % (
                    cur_segment_len,
                    int(cur_segment_len * 0.95),
                    segment_num,
                    int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
                ))
                cur_segment_len = int(cur_segment_len * 0.95)
                segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)

            append_emb = append_emb[init_cur_segment_len - cur_segment_len:]
        if model_args.trunc_type == "right":
            complete_emb = np.concatenate((first_emb, append_emb), axis=0)
        else:
            complete_emb = np.concatenate((append_emb, first_emb), axis=0)
        print("seq len: %d, seq embedding matrix len: %d" % (ori_seq_len, complete_emb.shape[0] + (2 if model_args.matrix_add_special_token else 0)))
        print("-" * 50)
        assert complete_emb.shape[0] == ori_seq_len
        if model_args.matrix_add_special_token:
            complete_emb = np.concatenate((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), axis=0)
        init_emb = complete_emb
    return init_emb


def predict_embedding(
        sample,
        trunc_type,
        embedding_type,
        repr_layers=[-1],
        truncation_seq_length=4094,
        device=None,
        version="600M",
        matrix_add_special_token=False,
        fp16=False,
        token=None
):
    global esmc_global_model, esmc_global_version
    assert "bos" in embedding_type \
           or "representations" in embedding_type \
           or "matrix" in embedding_type \
           or "vector" in embedding_type \
           or "contacts" in embedding_type

    if len(sample) > 2:
        protein_id, protein_seq = sample[0], sample[2]
    else:
        protein_id, protein_seq = sample[0], sample[1]

    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    # protein_seq = clean_seq_esm(protein_id, protein_seq)
    protein_seq = ESMProtein(sequence=protein_seq)

    if esmc_global_model is None \
            or esmc_global_version is None \
            or esmc_global_version != version:

        if version == "6B":
            try:
                llm_name = "esmc-6b-2024-12"
                esmc_global_model = ESM3ForgeInferenceClient(
                    model="esmc-6b-2024-12",
                    url="https://forge.evolutionaryscale.ai",
                    token=token
                )
            except Exception as e:
                print(e)
                raise Exception("access to ESMC-6B API is denied. Check access token or network connection.")
        elif version == "600M":
            llm_name = "esmc_600m"
            esmc_global_model = ESMC.from_pretrained("esmc_600m")
        elif version == "300M":
            llm_name = "esmc_300m"
            esmc_global_model = ESMC.from_pretrained("esmc_300m")
        else:
            raise Exception("not support this version=%s" % version)
        print("LLM: %s, version: %s, device: %s" % (llm_name, version, str(device)))
        esmc_global_version = version

    if fp16:
        esmc_global_model.half()

    if device is None:
        device = next(esmc_global_model.parameters()).device
    else:
        model_device = next(esmc_global_model.parameters()).device
        if device != model_device:
            esmc_global_model = esmc_global_model.to(device)
    embeddings = {}
    with torch.no_grad():
        try:
            protein_tensor = esmc_global_model.encode(protein_seq)
            logits_output = esmc_global_model.logits(
                protein_tensor, LogitsConfig(return_embeddings=True)
            )
            truncate_len = min(truncation_seq_length, len(protein_seq))
            processed_seq_len = truncate_len + 2
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_token:
                    embedding = logits_output.embeddings.to(device="cpu")[0, 0: truncate_len + 2, :].clone().numpy()
                else:
                    embedding = logits_output.embeddings.to(device="cpu")[0, 1: truncate_len + 1, :].clone().numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = logits_output.embeddings.to(device="cpu")[0, 0, :].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                raise Exception("embedding_type 'contacts' has not been supported yet.")
            if len(embeddings) > 1:
                return embeddings, processed_seq_len
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], processed_seq_len
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {protein_id} of length {len(protein_seq)}.")
                print("Please reduce the 'truncation_seq_length'")
            # raise Exception(e)
    return None, None


def get_args():
    parser = argparse.ArgumentParser(description='ESMC Embedding')
    # for one seq
    parser.add_argument("--seq_id", type=str, default=None,
                        help="the seq id")
    parser.add_argument("--seq", type=str, default=None,
                        help="when to input a seq")
    parser.add_argument("--seq_type", type=str, default="prot",
                        choices=["prot", "multi_prot"],
                        help="the input seq type")

    # for many
    parser.add_argument("--input_file", type=str, default=None,
                        help="the input file（format: fasta or csv or tsv)")
    # for input csv
    parser.add_argument("--id_idx", type=int, default=None,
                        help="id col idx(0 start)")
    parser.add_argument("--seq_idx", type=int, default=None,
                        help="seq col idx(0 start)")

    # for saved path
    parser.add_argument("--save_path", type=str, default=None,
                        help="embedding file save dir path")

    # for trained llm
    parser.add_argument("--llm_type", type=str, default="esmc",
                        choices=["esmc", "ESMC"],
                        help="llm type")
    parser.add_argument("--llm_version", type=str, default="600M",
                        choices=["300M", "600M", "6B"],
                        help="llm version")

    # for embedding
    parser.add_argument("--embedding_type", type=str, default="matrix",
                        choices=["matrix", "vector", "contact"],
                        help="llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right",
                        choices=["left", "right"],
                        help="llm trunc type of seq.")
    parser.add_argument("--truncation_seq_length", type=int,
                        default=4094,
                        help="truncation seq length.")
    parser.add_argument("--matrix_add_special_token", action="store_true",
                        help="whether to add special token embedding in seq representation matrix")
    parser.add_argument("--embedding_complete",  action="store_true",
                        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")
    parser.add_argument("--embedding_complete_seg_overlap",  action="store_true",
                        help="segment overlap")
    parser.add_argument("--embedding_fixed_len_a_time", type=int, default=None,
                        help="the embedding fixed length of once inference for longer sequence")

    parser.add_argument("--access_token", type=str, default=None, help="ESMC-6B Forge API access token.")

    # for running
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="the gpu id to use.")
    input_args = parser.parse_args()
    return input_args


def main(args):
    if args.gpu_id >= 0:
        gpu_id = args.gpu_id
    else:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        print("gpu_id: ", gpu_id)
    """
    if gpu_id is None or gpu_id == -1:
        args.device = None
    else:
        args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    """
    args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # esm_global_model.to(args.device)
    assert (args.input_file is not None and os.path.exists(args.input_file)) or args.seq is not None
    print("input seq type: %s" % args.seq_type)
    print("args device: %s" % args.device)
    embedding_type = args.embedding_type
    seq_type = args.seq_type
    emb_save_path = args.save_path
    print("emb save dir: %s" % os.path.abspath(emb_save_path))
    if seq_type not in ["prot", "multi_prot"]:
        print("Error! arg: --seq_type=%s is not 'prot' or 'multi_prot'" % seq_type)
        sys.exit(-1)
    input_seq_type = seq_type
    if seq_type == "multi_prot":
        seq_type = "prot"
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)

    if args.input_file:
        done = 0
        file_reader = fasta_reader
        if args.input_file.endswith(".csv"):
            file_reader = csv_reader
        elif args.input_file.endswith(".tsv"):
            file_reader = tsv_reader

        for row in file_reader(args.input_file):
            if args.id_idx is None or args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[args.id_idx].strip(), row[args.seq_idx].upper()
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            if not os.path.exists(embedding_filepath):
                emb_list = []
                # multi seqs
                if "," in seq:
                    multi_seqs = [v.strip().upper() for v in seq.split(",")]
                else:
                    multi_seqs = [seq.strip().upper()]
                assert len(multi_seqs) == 1 or input_seq_type in ["multi_prot"]
                for seq in multi_seqs:
                    input_seq_len = len(seq)
                    if args.embedding_complete:
                        truncation_seq_length = input_seq_len
                    else:
                        truncation_seq_length = min(input_seq_len, args.truncation_seq_length)
                    while True:
                        # 设置了一次性推理长度
                        if args.embedding_fixed_len_a_time and args.embedding_fixed_len_a_time > 0:
                            emb, processed_seq_len = predict_embedding(
                                [seq_id, seq_type, seq],
                                args.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=args.embedding_fixed_len_a_time,
                                device=args.device,
                                version=args.llm_version,
                                matrix_add_special_token=args.matrix_add_special_token,
                                token=args.access_token,
                            )
                            use_cpu = False
                            if emb is None:
                                emb, processed_seq_len = predict_embedding(
                                    [seq_id, seq_type, seq],
                                    args.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=args.embedding_fixed_len_a_time,
                                    device=torch.device("cpu"),
                                    version=args.llm_version,
                                    matrix_add_special_token=args.matrix_add_special_token,
                                    token=args.access_token,
                                )
                                use_cpu = True
                            # embedding全
                            if emb is not None and input_seq_len > args.embedding_fixed_len_a_time:
                                emb = complete_embedding_matrix(
                                    seq_id,
                                    seq_type,
                                    seq,
                                    truncation_seq_length,
                                    emb,
                                    args,
                                    embedding_type,
                                    use_cpu=use_cpu,
                                    token=args.access_token,
                                )
                            if use_cpu:
                                print("use_cpu: %r" % use_cpu)
                        else:
                            emb, processed_seq_len = predict_embedding(
                                [seq_id, seq_type, seq],
                                args.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=args.truncation_seq_length,
                                device=args.device,
                                version=args.llm_version,
                                matrix_add_special_token=args.matrix_add_special_token,
                                token=args.access_token,
                            )
                            use_cpu = False
                            if emb is None:
                                emb, processed_seq_len = predict_embedding(
                                    [seq_id, seq_type, seq],
                                    args.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=args.truncation_seq_length,
                                    device=torch.device("cpu"),
                                    version=args.llm_version,
                                    matrix_add_special_token=args.matrix_add_special_token,
                                    token=args.access_token,
                                )
                                use_cpu = True
                            # embedding全
                            if emb is not None and input_seq_len > truncation_seq_length:
                                emb = complete_embedding_matrix(
                                    seq_id,
                                    seq_type,
                                    seq,
                                    truncation_seq_length,
                                    emb,
                                    args,
                                    embedding_type,
                                    use_cpu=use_cpu,
                                    token=args.access_token,
                                )
                            if use_cpu:
                                print("use_cpu: %r" % use_cpu)
                        if emb is not None:
                            # print("seq_len: %d" % len(seq))
                            # print("emb shape:", embedding_info.shape)
                            # torch.save(emb, embedding_filepath)
                            emb_list.append(emb)
                            break
                        print("%s embedding error, max_len from %d truncate to %d" % (
                            seq_id, truncation_seq_length,int(truncation_seq_length * 0.95)
                        ))
                        truncation_seq_length = int(truncation_seq_length * 0.95)
                if input_seq_type in ["multi_prot"]:
                    torch.save(emb_list, embedding_filepath)
                else:
                    torch.save(emb_list[0], embedding_filepath)
                torch.cuda.empty_cache()
            else:
                print("%s exists." % embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)
        print("embedding over, done: %d" % done)
    elif args.seq:
        print("input seq length: %d" % len(args.seq))
        emb, processed_seq_len = predict_embedding(
            [args.seq_id, seq_type, args.seq],
            args.trunc_type,
            embedding_type,
            repr_layers=[-1],
            truncation_seq_length=args.truncation_seq_length,
            device=args.device,
            version=args.llm_version,
            matrix_add_special_token=args.matrix_add_special_token,
            token=args.access_token,
        )
        print("done seq length: %d" % processed_seq_len)
        print(emb)
        if emb is not None:
            print(emb.shape)


if __name__ == "__main__":
    run_args = get_args()
    main(run_args)