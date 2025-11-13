#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/10 20:18
@project: LucaVirusTasks
@file: predict
@desc: predict or inference for trained downstream models
'''
import csv
import json
import uuid
import os, sys
import torch
import codecs
import time, shutil
import numpy as np
import argparse
from collections import OrderedDict
from subword_nmt.apply_bpe import BPE
from transformers import BertConfig
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, device_memory, available_gpu_id, load_labels, seq_type_is_match_seq,\
        download_trained_checkpoint_lucaone, download_trained_checkpoint_lucavirus, \
        download_trained_checkpoint_downstream_tasks
    from common.multi_label_metrics import relevant_indexes
    from encoder import Encoder
    from batch_converter import BatchConverter
    from common.alphabet import Alphabet
    from file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from common.luca_base import LucaBase
    from lucapair.models.LucaPairHomo import LucaPairHomo
    from lucapair.models.LucaPairHeter import LucaPairHeter
    from lucapair.models.LucaPairDecoderAB import LucaPairDecoderAB
    from lucapair.models.LucaPairDecoderDual import LucaPairDecoderDual
    from lucapair.models.LucaPairEncoderAB import LucaPairEncoderAB
    from lucapair.models.LucaPairEncoderDual import LucaPairEncoderDual
    from lucapair.models.LucaPairIntraInter import LucaPairIntraInter
    from lucatriple.models.LucaTripleHomo import LucaTripleHomo
    from lucatriple.models.LucaTripleHeter import LucaTripleHeter
except ImportError:
    from src.utils import to_device, device_memory, available_gpu_id, load_labels, seq_type_is_match_seq, \
        download_trained_checkpoint_lucaone, download_trained_checkpoint_lucavirus, \
        download_trained_checkpoint_downstream_tasks
    from src.common.multi_label_metrics import relevant_indexes
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter
    from src.common.alphabet import Alphabet
    from src.file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from src.common.luca_base import LucaBase
    from lucapair.models.LucaPairHomo import LucaPairHomo
    from lucapair.models.LucaPairHeter import LucaPairHeter
    from lucapair.models.LucaPairDecoderAB import LucaPairDecoderAB
    from lucapair.models.LucaPairDecoderDual import LucaPairDecoderDual
    from lucapair.models.LucaPairEncoderAB import LucaPairEncoderAB
    from lucapair.models.LucaPairEncoderDual import LucaPairEncoderDual
    from lucapair.models.LucaPairIntraInter import LucaPairIntraInter
    from lucatriple.models.LucaTripleHomo import LucaTripleHomo
    from lucatriple.models.LucaTripleHeter import LucaTripleHeter



def transform_one_sample_2_feature(
        device,
        input_mode,
        encoder,
        batch_convecter,
        row
):
    """
    tranform the input to features
    :param device:
    :param input_mode:
    :param encoder:
    :param batch_convecter:
    :param row:
    :return:
    """
    batch_info = []
    if input_mode in ["triple"]:
        # seq_id_a, seq_id_b, seq_id_c, seq_type_a, seq_type_b, seq_type_c, seq_a, seq_b, seq_c
        en = encoder.encode_triple(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            vector_filename_a=None,
            vector_filename_b=None,
            vector_filename_c=None,
            matrix_filename_a=None,
            matrix_filename_b=None,
            matrix_filename_c=None,
            label=None
        )
        en_list = en
        batch_info.append([row[0], row[1], row[2], row[6], row[7], row[8]])
        seq_lens = [len(row[6]), len(row[7]), len(row[8])]
    elif input_mode in ["pair"]:
        # seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
        en = encoder.encode_pair(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            vector_filename_a=None,
            vector_filename_b=None,
            matrix_filename_a=None,
            matrix_filename_b=None,
            label=None
        )
        en_list = en
        batch_info.append([row[0], row[1], row[4], row[5]])
        seq_lens = [len(row[4]), len(row[5])]
    else:
        # seq_id, seq_type, seq
        en = encoder.encode_single(
            row[0],
            row[1],
            row[2],
            vector_filename=None,
            matrix_filename=None,
            label=None
        )
        en_list = en
        batch_info.append([row[0], row[2]])
        seq_lens = len(row[2])
    batch = [en_list]
    if isinstance(batch[0], list):
        batch_features = []
        for cur_batch in batch[0]:
            cur_batch_features = batch_convecter([cur_batch])
            cur_batch_features, cur_sample_num = to_device(device, cur_batch_features)
            batch_features.append(cur_batch_features)
    else:
        batch_features = batch_convecter(batch)
        batch_features, cur_sample_num = to_device(device, batch_features)
    return batch_info, batch_features, [seq_lens]


def predict_probs(
        args,
        encoder,
        batch_convecter,
        model,
        row
):
    """
    predict the prob
    :param args:
    :param encoder:
    :param batch_convecter:
    :param model:
    :param row:
    :return:
    """
    batch_info, batch_features, seq_lens = transform_one_sample_2_feature(
        args.device,
        args.input_mode,
        encoder,
        batch_convecter,
        row
    )
    if isinstance(batch_features, list):
        probs = []
        for cur_batch_features in batch_features:
            cur_probs = model(**cur_batch_features)[1]
            if cur_probs.is_cuda:
                cur_probs = cur_probs.detach().cpu().numpy()
            else:
                cur_probs = cur_probs.detach().numpy()
            probs.append(cur_probs)
    else:
        probs = model(**batch_features)[1]
        if probs.is_cuda:
            probs = probs.detach().cpu().numpy()
        else:
            probs = probs.detach().numpy()
    return batch_info, probs, seq_lens


def predict_token_level_binary_class(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    """
    prediction for the token-level binary classification
    :param args:
    :param encoder:
    :param batch_convecter:
    :param label_id_2_name:
    :param model:
    :param row:
    :return:
    """
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # probs: (batch_size, seq_len, 1)
    # print("probs dim: ", probs.ndim)
    # preds: (batch_size, seq_len, 1)
    preds = (probs >= args.threshold).astype(int)
    probs = probs.tolist()
    preds = preds.tolist()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "triple":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                info[6],
                [v[0] for v in probs[idx]],
                [v[0] for v in preds[idx]],
                [label_id_2_name[v[0]] for v in preds[idx]]
            ]
            if len(info) > 6:
                cur_res += info[6:]
        elif args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                [v[0] for v in probs[idx]],
                [v[0] for v in preds[idx]],
                [label_id_2_name[v[0]] for v in preds[idx]]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                [v[0] for v in probs[idx]],
                [v[0] for v in preds[idx]],
                [label_id_2_name[v[0]] for v in preds[idx]]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_token_level_multi_class(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row,
        topk=5
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # probs: (batch_size, seq_len, label_size)
    # preds: (batch_size, seq_len)
    preds = np.argmax(probs, axis=-1)
    preds = preds.tolist()
    res = []
    if topk is not None and topk > 1:
        # print("topk: %d" % topk)
        # preds_topk: (batch_size, seq_len, topk)
        preds_topk = np.argsort(probs, axis=-1)[:, ::-1][:, :topk]
        probs = probs.tolist()
        preds_topk = preds_topk.tolist()
        for idx, info in enumerate(batch_info):
            cur_seq_topk_probs = []
            cur_seq_topk_label_indices = []
            cur_seq_topk_labels = []
            for token_idx in range(len(preds_topk[idx])):
                cur_token_topk_probs = []
                cur_token_topk_label_indices = []
                cur_token_topk_labels = []
                for label_idx in preds_topk[idx][token_idx]:
                    cur_token_topk_probs.append(probs[idx][token_idx][label_idx])
                    cur_token_topk_label_indices.append(label_idx)
                    cur_token_topk_labels.append(label_id_2_name[label_idx])
                cur_seq_topk_probs.append(cur_token_topk_probs)
                cur_seq_topk_label_indices.append(cur_token_topk_label_indices)
                cur_seq_topk_labels.append(cur_token_topk_labels)
            if args.input_mode == "triple":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    info[4],
                    info[5],
                    [probs[idx][token_idx][label_idx] for token_idx, label_idx in enumerate(preds[idx])],
                    [label_idx for label_idx in preds[idx]],
                    [label_id_2_name[label_idx] for label_idx in preds[idx]],
                    cur_seq_topk_probs,
                    cur_seq_topk_label_indices,
                    cur_seq_topk_labels
                ]
                if len(info) > 6:
                    cur_res += info[6:]
            elif args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    [probs[idx][token_idx][label_idx] for token_idx, label_idx in enumerate(preds[idx])],
                    [label_idx for label_idx in preds[idx]],
                    [label_id_2_name[label_idx] for label_idx in preds[idx]],
                    cur_seq_topk_probs,
                    cur_seq_topk_label_indices,
                    cur_seq_topk_labels
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    [probs[idx][token_idx][label_idx] for token_idx, label_idx in enumerate(preds[idx])],
                    [label_idx for label_idx in preds[idx]],
                    [label_id_2_name[label_idx] for label_idx in preds[idx]],
                    cur_seq_topk_probs,
                    cur_seq_topk_label_indices,
                    cur_seq_topk_labels
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res
    else:
        res = []
        probs = probs.tolist()
        for idx, info in enumerate(batch_info):
            if args.input_mode == "triple":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    info[4],
                    info[5],
                    [probs[idx][token_idx][label_idx] for token_idx, label_idx in enumerate(preds[idx])],
                    [label_idx for label_idx in preds[idx]],
                    [label_id_2_name[label_idx] for label_idx in preds[idx]]
                ]
                if len(info) > 6:
                    cur_res += info[6:]
            elif args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    [probs[idx][token_idx][label_idx] for token_idx, label_idx in enumerate(preds[idx])],
                    [label_idx for label_idx in preds[idx]],
                    [label_id_2_name[label_idx] for label_idx in preds[idx]]
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    [probs[idx][token_idx][label_idx] for token_idx, label_idx in enumerate(preds[idx])],
                    [label_idx for label_idx in preds[idx]],
                    [label_id_2_name[label_idx] for label_idx in preds[idx]],
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res


def predict_token_level_multi_label(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    # probs: (batch_size, seq_len, label_size)
    # preds: (batch_size, seq_len, size of the prob > threshold)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    probs = probs.tolist()
    preds = preds.tolist()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "triple":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                info[5],
                [[probs[idx][token_idx][label_idx] for label_idx in preds[idx][token_idx]] for token_idx in range(len(preds[idx]))],
                [[label_id_2_name[label_idx] for label_idx in pred] for pred in preds[idx]],
                [[label_id_2_name[label_idx] for label_idx in pred] for pred in preds[idx]]
            ]
            if len(info) > 6:
                cur_res += info[6:]
        elif args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                [[probs[idx][token_idx][label_idx] for label_idx in preds[idx][token_idx]] for token_idx in range(len(preds[idx]))],
                [[label_id_2_name[label_idx] for label_idx in pred] for pred in preds[idx]],
                [[label_id_2_name[label_idx] for label_idx in pred] for pred in preds[idx]]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                [[probs[idx][token_idx][label_idx] for label_idx in preds[idx][token_idx]] for token_idx in range(len(preds[idx]))],
                [[label_id_2_name[label_idx] for label_idx in pred] for pred in preds[idx]],
                [[label_id_2_name[label_idx] for label_idx in pred] for pred in preds[idx]]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_token_level_regression(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # probs: (batch_size, seq_len, 1)
    probs = probs.tolist()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "triple":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                info[5],
                [prob[0] for prob in probs[idx]],
                [prob[0] for prob in probs[idx]],
                [prob[0] for prob in probs[idx]]
            ]
            if len(info) > 6:
                cur_res += info[6:]
        elif args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                [prob[0] for prob in probs[idx]],
                [prob[0] for prob in probs[idx]],
                [prob[0] for prob in probs[idx]]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                [prob[0] for prob in probs[idx]],
                [prob[0] for prob in probs[idx]],
                [prob[0] for prob in probs[idx]]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_binary_class(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    # probs: (batch_size, 1)
    # preds: (batch_size, 1)
    preds = (probs >= args.threshold).astype(int)
    probs = probs.tolist()
    preds = preds.tolist()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "triple":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                info[5],
                probs[idx][0],
                preds[idx][0],
                label_id_2_name[preds[idx][0]]
            ]
            if len(info) > 6:
                cur_res += info[6:]
        elif args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                probs[idx][0],
                preds[idx][0],
                label_id_2_name[preds[idx][0]]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                probs[idx][0],
                preds[idx][0],
                label_id_2_name[preds[idx][0]]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_multi_class(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row,
        topk=5
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    # probs: (batch_size, label_size)
    # preds: (batch_size, )
    # preds_topk: (batch_size, topk)
    preds = np.argmax(probs, axis=-1)
    preds = preds.tolist()
    res = []
    if topk is not None and topk > 1:
        # print("topk: %d" % topk)
        preds_topk = np.argsort(probs, axis=-1)[:, ::-1][:, :topk]
        probs = probs.tolist()
        preds_topk = preds_topk.tolist()
        for idx, info in enumerate(batch_info):
            cur_topk_probs = []
            cur_topk_label_indices = []
            cur_topk_labels = []
            for label_idx in preds_topk[idx]:
                cur_topk_probs.append(probs[idx][label_idx])
                cur_topk_label_indices.append(label_idx)
                cur_topk_labels.append(label_id_2_name[label_idx])
            if args.input_mode == "triple":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    info[4],
                    info[5],
                    probs[idx][preds[idx]],
                    preds[idx],
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_label_indices,
                    cur_topk_labels
                ]
                if len(info) > 6:
                    cur_res += info[6:]
            elif args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    probs[idx][preds[idx]],
                    preds[idx],
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_label_indices,
                    cur_topk_labels
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    probs[idx][preds[idx]],
                    preds[idx],
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_label_indices,
                    cur_topk_labels
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res
    else:
        probs = probs.tolist()
        res = []
        for idx, info in enumerate(batch_info):
            if args.input_mode == "triple":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    info[4],
                    info[5],
                    probs[idx][preds[idx]],
                    preds[idx],
                    label_id_2_name[preds[idx]]
                ]
                if len(info) > 6:
                    cur_res += info[6:]
            elif args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    probs[idx][preds[idx]],
                    preds[idx],
                    label_id_2_name[preds[idx]]
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    probs[idx][preds[idx]],
                    preds[idx],
                    label_id_2_name[preds[idx]]
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res


def predict_seq_level_multi_label(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    # probs: (batch_size, label_size)
    # preds: (batch_size, size of the prob > threshold)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    probs = probs.tolist()
    preds = preds.tolist()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "triple":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                info[5],
                [probs[idx][label_index] for label_index in preds[idx]],
                [label_index for label_index in preds[idx]],
                [label_id_2_name[label_index] for label_index in preds[idx]]
            ]
            if len(info) > 6:
                cur_res += info[6:]
        elif args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                [probs[idx][label_index] for label_index in preds[idx]],
                [label_index for label_index in preds[idx]],
                [label_id_2_name[label_index] for label_index in preds[idx]]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                [probs[idx][label_index] for label_index in preds[idx]],
                [label_index for label_index in preds[idx]],
                [label_id_2_name[label_index] for label_index in preds[idx]]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_regression(
        args,
        encoder,
        batch_convecter,
        label_id_2_name,
        model,
        row
):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # probs: (batch_size, 1)
    probs = probs.tolist()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "triple":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                info[5],
                probs[idx][0],
                probs[idx][0],
                probs[idx][0]
            ]
            if len(info) > 6:
                cur_res += info[6:]
        elif args.input_mode == "pair":
            cur_res = [
                info[0],
                info[1],
                info[2],
                info[3],
                probs[idx][0],
                probs[idx][0],
                probs[idx][0]
            ]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [
                info[0],
                info[1],
                probs[idx][0],
                probs[idx][0],
                probs[idx][0]
            ]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def load_tokenizer(
        args,
        model_dir,
        seq_tokenizer_class
):
    seq_subword, seq_tokenizer = None, None
    if not hasattr(args, "has_seq_encoder") or args.has_seq_encoder:
        # 是否分词
        if args.seq_subword:
            if os.path.exists(os.path.join(model_dir, "sequence")):
                seq_tokenizer = seq_tokenizer_class.from_pretrained(
                    os.path.join(model_dir, "sequence"),
                    do_lower_case=args.do_lower_case
                )
            else:
                seq_tokenizer = seq_tokenizer_class.from_pretrained(
                    os.path.join(model_dir, "tokenizer"),
                    do_lower_case=args.do_lower_case
                )
            bpe_codes = codecs.open(args.codes_file)
            seq_subword = BPE(bpe_codes, merges=-1, separator='')
        else:
            seq_subword = None
            seq_tokenizer = seq_tokenizer_class.from_predefined(args.seq_vocab_path)
            if args.not_prepend_bos:
                seq_tokenizer.prepend_bos = False
            if args.not_append_eos:
                seq_tokenizer.append_eos = False
    return seq_subword, seq_tokenizer


def load_trained_model(
        model_config,
        args,
        model_class,
        model_dirpath
):
    # load exists checkpoint
    print("load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        print(e)
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(
            os.path.join(model_dirpath, "pytorch.pth"),
            map_location=torch.device("cpu")
        )
        model_state_dict_keys = set()
        for key in model.state_dict():
            model_state_dict_keys.add(key)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        # print("diff:")
        # print(model_state_dict_keys.difference(new_state_dict.keys()))
        model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()
    return model


def load_model(
        args,
        model_name,
        model_dir
):
    # load tokenizer and model
    begin_time = time.time()
    device = torch.device(args.device)
    print("load model(%s) on cuda:" % model_name, device)

    if args.model_type in ["lucapair_homo"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairHomo
    elif args.model_type in ["lucapair_heter"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairHeter
    elif args.model_type in ["lucapair_decoder_ab"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairDecoderAB
    elif args.model_type in ["lucapair_decoder_dual"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairDecoderDual
    elif args.model_type in ["lucapair_encoder_ab"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairEncoderAB
    elif args.model_type in ["lucapair_encoder_dual"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairEncoderDual
    elif args.model_type in ["lucapair_intrainter"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPairIntraInter
    elif args.model_type in ["lucatriple_homo"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaTripleHomo
    elif args.model_type in ["lucatriple_heter"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaTripleHeter
    elif args.model_type in ["lucabase"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaBase
    else:
        raise Exception("Not support the model_type=%s" % args.model_type)
    seq_subword, seq_tokenizer = load_tokenizer(args, model_dir, seq_tokenizer_class)

    # config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r"), encoding="UTF-8"))
    model_config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))

    model = load_trained_model(model_config, args, model_class, model_dir)
    print("the time for loading model:", time.time() - begin_time)

    return model_config, seq_subword, seq_tokenizer, model


def create_encoder_batch_convecter(
        model_args,
        seq_subword,
        seq_tokenizer
):
    """
    create encoder and batch convecter for inputs
    :param model_args:
    :param seq_subword:
    :param seq_tokenizer:
    :return:
    """
    if hasattr(model_args, "input_mode") and model_args.input_mode in ["pair"]:
        assert model_args.seq_max_length is not None \
               or (model_args.seq_max_length_a is not None and model_args.seq_max_length_b is not None)
        if model_args.seq_max_length is None:
            model_args.seq_max_length = max(model_args.seq_max_length_a, model_args.seq_max_length_b)
    else:
        assert model_args.seq_max_length is not None
    encoder_config = {
        "llm_type": model_args.llm_type,
        "llm_version": model_args.llm_version,
        "llm_step": model_args.llm_step,
        "llm_dirpath": model_args.llm_dirpath,
        "input_type": model_args.input_type,
        "trunc_type": model_args.trunc_type,
        "seq_max_length": model_args.seq_max_length,
        "atom_seq_max_length": None,
        "vector_dirpath": model_args.vector_dirpath,
        "matrix_dirpath": model_args.matrix_dirpath,
        "matrix_add_special_token": model_args.matrix_add_special_token,
        "local_rank": model_args.gpu_id,
        "max_sentence_length": model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
        "max_sentences": model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
        "embedding_complete": model_args.embedding_complete,
        "embedding_complete_seg_overlap": model_args.embedding_complete_seg_overlap,
        "embedding_fixed_len_a_time": model_args.embedding_fixed_len_a_time,
        "matrix_embedding_exists": model_args.matrix_embedding_exists,
        "use_cpu": True if model_args.gpu_id < 0 else False,
        "buffer_size": model_args.buffer_size
    }
    encoder = Encoder(**encoder_config)

    batch_converter = BatchConverter(
        task_level_type=model_args.task_level_type,
        label_size=model_args.label_size,
        output_mode=model_args.output_mode,
        seq_subword=seq_subword,
        seq_tokenizer=seq_tokenizer,
        no_position_embeddings=model_args.no_position_embeddings,
        no_token_type_embeddings=model_args.no_token_type_embeddings,
        truncation_seq_length=model_args.truncation_seq_length if hasattr(model_args, "truncation_seq_length") else model_args.seq_max_length,
        truncation_matrix_length=model_args.truncation_matrix_length if hasattr(model_args, "truncation_matrix_length") else model_args.matrix_max_length,
        trunc_type=model_args.trunc_type if hasattr(model_args, "trunc_type") else "right",
        atom_tokenizer=None,
        atom_truncation_seq_length=None,
        atom_truncation_matrix_length=None,
        padding_idx=0,
        unk_idx=1,
        cls_idx=2,
        eos_idx=3,
        mask_idx=4,
        ignore_index=model_args.ignore_index,
        non_ignore=model_args.non_ignore,
        prepend_bos=not model_args.not_prepend_bos,
        append_eos=not model_args.not_append_eos,
        max_sentence_length=model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
        max_sentences=model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
        matrix_add_special_token=model_args.matrix_add_special_token if hasattr(model_args, "matrix_add_special_token") else False
    )
    return encoder, batch_converter


# global
global_model_config, global_seq_subword, global_seq_tokenizer, global_trained_model = None, None, None, None


def run(
        sequences,
        llm_truncation_seq_length,
        model_path,
        dataset_name,
        dataset_type,
        task_type,
        task_level_type,
        model_type,
        input_type,
        input_mode,
        time_str,
        step,
        gpu_id,
        threshold,
        topk,
        emb_dir,
        matrix_embedding_exists
):
    global global_model_config, global_seq_subword, global_seq_tokenizer, global_trained_model
    model_dir = "%s/models/%s/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type,
        time_str, step if step == "best" else "checkpoint-{}".format(step))
    config_dir = "%s/logs/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type,  time_str
    )

    model_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    print("-" * 25 + "Trained Model Args" + "-" * 25)
    print(model_args.__dict__)
    print("-" * 50)
    model_args.llm_truncation_seq_length = llm_truncation_seq_length
    model_args.seq_max_length = llm_truncation_seq_length
    model_args.atom_seq_max_length = None # to do
    model_args.truncation_seq_length = model_args.seq_max_length
    model_args.truncation_matrix_length = model_args.matrix_max_length

    model_args.matrix_embedding_exists = matrix_embedding_exists
    # embedding saved dir during prediction
    if emb_dir and not matrix_embedding_exists:
        # now = datetime.now()
        # formatted_time = now.strftime("%Y%m%d%H%M%S")
        unique_string = str(uuid.uuid4())
        emb_dir = os.path.join(emb_dir, "%s-%d" % (unique_string, gpu_id))
    model_args.emb_dir = emb_dir
    model_args.vector_dirpath = model_args.emb_dir if model_args.emb_dir else None
    model_args.matrix_dirpath = model_args.emb_dir if model_args.emb_dir else None

    model_args.dataset_name = dataset_name
    model_args.dataset_type = dataset_type
    model_args.task_type = task_type
    model_args.task_level_type = task_level_type
    model_args.model_type = model_type
    model_args.input_type = input_type
    model_args.time_str = time_str
    model_args.step = step
    model_args.gpu_id = gpu_id

    if not hasattr(model_args, "embedding_complete"):
        model_args.embedding_complete = False
    if not hasattr(model_args, "embedding_complete_seg_overlap"):
        model_args.embedding_complete_seg_overlap = False
    if not hasattr(model_args, "embedding_fixed_len_a_time"):
        model_args.embedding_fixed_len_a_time = None
    if not hasattr(model_args, "matrix_add_special_token"):
        model_args.matrix_add_special_token = False

    if not hasattr(model_args, "non_ignore"):
        model_args.non_ignore = True
    model_args.threshold = threshold

    if model_args.label_filepath:
        model_args.label_filepath = model_args.label_filepath.replace("../", "%s/" % model_path)
    if not os.path.exists(model_args.label_filepath):
        model_args.label_filepath = os.path.join(config_dir, "label.txt")

    if gpu_id is None or gpu_id < 0:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        model_args.gpu_id = gpu_id
    print("------Before loading the model:------")
    print("GPU ID: %d" % gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    device_memory(None if gpu_id == -1 else gpu_id)

    # Step2: loading the tokenizer and model
    if global_trained_model is None or next(global_trained_model.parameters()).device != model_args.device:
        global_trained_model = None
        model_config, seq_subword, seq_tokenizer, trained_model = load_model(model_args, model_type, model_dir)
        global_model_config = model_config
        global_seq_subword = seq_subword
        global_seq_tokenizer = seq_tokenizer
        global_trained_model = trained_model
    else:
        model_config = global_model_config
        seq_subword = global_seq_subword
        seq_tokenizer = global_seq_tokenizer
        trained_model = global_trained_model

    print("------After loaded the model:------")
    device_memory(None if gpu_id == -1 else gpu_id)
    encoder, batch_convecter = create_encoder_batch_convecter(model_args, seq_subword, seq_tokenizer)

    # V2的不同点：embedding in advance
    # embedding in advance
    print("matrix_embedding_exists: %r, gpu_id: %d" % (matrix_embedding_exists, gpu_id))
    if not matrix_embedding_exists and gpu_id > -1:
        # model先to cpu
        trained_model.to(torch.device("cpu"))
        assert model_args.emb_dir is not None
        if not os.path.exists(model_args.emb_dir):
            os.makedirs(model_args.emb_dir)
        encoder.save_emb_to_disk = True
        for item in sequences:
            if input_mode == "triple":
                seq_id_a = item[0]
                seq_type_a = item[3]
                seq_a = item[6]
                seq_id_b = item[1]
                seq_type_b = item[4]
                seq_b = item[7]
                seq_id_c = item[2]
                seq_type_c = item[5]
                seq_c = item[8]
                encoder.__get_embedding__(
                    seq_id=seq_id_a,
                    seq_type=seq_type_a,
                    seq=seq_a,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
                encoder.__get_embedding__(
                    seq_id=seq_id_b,
                    seq_type=seq_type_b,
                    seq=seq_b,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
                encoder.__get_embedding__(
                    seq_id=seq_id_c,
                    seq_type=seq_type_c,
                    seq=seq_c,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
            elif input_mode == "pair":
                seq_id_a = item[0]
                seq_type_a = item[2]
                seq_a = item[4]
                seq_id_b = item[1]
                seq_type_b = item[3]
                seq_b = item[5]
                encoder.__get_embedding__(
                    seq_id=seq_id_a,
                    seq_type=seq_type_a,
                    seq=seq_a,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
                encoder.__get_embedding__(
                    seq_id=seq_id_b,
                    seq_type=seq_type_b,
                    seq=seq_b,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
            else:
                seq_id = item[0]
                seq_type = item[1]
                seq = item[2]
                encoder.__get_embedding__(
                    seq_id=seq_id,
                    seq_type=seq_type,
                    seq=seq,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
            torch.cuda.empty_cache()
        encoder.matrix_embedding_exists = True

        # embedding 完之后to device
        trained_model.to(model_args.device)

    label_list = load_labels(model_args.label_filepath)
    label_id_2_name = {idx: name for idx, name in enumerate(label_list)}

    # Step 3: prediction
    if model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["binary_class", "binary-class"]:
        predict_func = predict_seq_level_binary_class
    elif model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
        predict_func = predict_seq_level_multi_class
    elif model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_label", "multi-label"]:
        predict_func = predict_seq_level_multi_label
    elif model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["regression"]:
        predict_func = predict_seq_level_regression
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["binary_class", "binary-class"]:
        predict_func = predict_token_level_binary_class
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
        predict_func = predict_token_level_multi_class
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["multi_label", "multi-label"]:
        predict_func = predict_token_level_multi_label
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["regression"]:
        predict_func = predict_token_level_regression
    else:
        raise Exception("the task_type=%s or task_level_type=%s error" % (task_type, model_args.task_level_type))

    predicted_results = []
    print()
    print("Device:", model_args.device)
    if hasattr(model_args, "input_mode") and model_args.input_mode in ["triple"]:
        for item in sequences:
            seq_id_a = item[0]
            seq_id_b = item[1]
            seq_id_c = item[2]
            seq_type_a = item[3]
            seq_type_b = item[4]
            seq_type_c = item[5]
            seq_a = item[6]
            seq_b = item[7]
            seq_c = item[8]
            record = [seq_id_a, seq_id_b, seq_id_c, seq_type_a, seq_type_b, seq_type_c, seq_a, seq_b, seq_c]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row=record,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([
                        seq_id_a, seq_id_b, seq_id_c, seq_a, seq_b, seq_c,
                        cur_res[0][6], cur_res[0][7],
                        cur_res[0][8], cur_res[0][9],
                        cur_res[0][10], cur_res[0][11]
                    ])
                else:
                    predicted_results.append([
                        seq_id_a, seq_id_b, seq_id_c, seq_a, seq_b, seq_c,
                        cur_res[0][6], cur_res[0][7], cur_res[0][8]
                    ])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row=record
                )
                predicted_results.append([
                    seq_id_a, seq_id_b, seq_id_c, seq_a, seq_b, seq_c,
                    cur_res[0][6], cur_res[0][7], cur_res[0][8]
                ])
    elif hasattr(model_args, "input_mode") and model_args.input_mode in ["pair"]:
        for item in sequences:
            seq_id_a = item[0]
            seq_id_b = item[1]
            seq_type_a = item[2]
            seq_type_b = item[3]
            seq_a = item[4]
            seq_b = item[5]
            record = [seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row=record,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([
                        seq_id_a, seq_id_b, seq_a, seq_b,
                        cur_res[0][4], cur_res[0][5],
                        cur_res[0][6], cur_res[0][7],
                        cur_res[0][8], cur_res[0][9]
                    ])
                else:
                    predicted_results.append([
                        seq_id_a, seq_id_b, seq_a, seq_b,
                        cur_res[0][4], cur_res[0][5], cur_res[0][6]
                    ])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row=record
                )
                predicted_results.append([
                    seq_id_a, seq_id_b, seq_a, seq_b,
                    cur_res[0][4], cur_res[0][5], cur_res[0][6]
                ])
    else:
        for item in sequences:
            seq_id = item[0]
            seq_type = item[1]
            seq = item[2]
            record = [seq_id, seq_type, seq]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                # print("task_level_type: %s, task_type: %s" % (task_level_type, task_type))
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row=record,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([
                        seq_id, seq,
                        cur_res[0][2], cur_res[0][3], cur_res[0][4],
                        cur_res[0][5], cur_res[0][6], cur_res[0][7]
                    ])
                else:
                    predicted_results.append([
                        seq_id, seq,
                        cur_res[0][2], cur_res[0][3], cur_res[0][4]
                    ])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row=record
                )
                predicted_results.append([
                    seq_id, seq,
                    cur_res[0][2], cur_res[0][3], cur_res[0][4]
                ])
    torch.cuda.empty_cache()
    # 删除embedding
    if not matrix_embedding_exists and os.path.exists(model_args.emb_dir) and input_type != "seq":
        shutil.rmtree(model_args.emb_dir)
    return predicted_results


def get_args():
    parser = argparse.ArgumentParser(description="Prediction")
    # for one seq sample of the input
    parser.add_argument("--seq_id", default=None, type=str,  help="the seq id")
    parser.add_argument("--seq_type", default=None, type=str, choices=["prot", "gene"], help="seq type.")
    parser.add_argument("--seq", default=None, type=str,  help="the sequence")

    # for one seq-seq sample of the input
    parser.add_argument("--seq_id_a", default=None, type=str,  help="the seq id a")
    parser.add_argument("--seq_type_a", default=None, type=str, choices=["prot", "gene"], help="seq type a.")
    parser.add_argument("--seq_a", default=None, type=str,  help="the sequence a")
    parser.add_argument("--seq_id_b", default=None, type=str,  help="the seq id b")
    parser.add_argument("--seq_type_b", default=None, type=str, choices=["prot", "gene"], help="seq type b.")
    parser.add_argument("--seq_b", default=None, type=str,  help="the sequence b")
    parser.add_argument("--seq_id_c", default=None, type=str,  help="the seq id c")
    parser.add_argument("--seq_type_c", default=None, type=str, choices=["prot", "gene"], help="seq type c.")
    parser.add_argument("--seq_c", default=None, type=str,  help="the sequence c")
    # for many samples
    parser.add_argument("--input_file", default=None, type=str, 
                        help="the fasta or csv format file for single-seq model,"
                             " or the csv format file for pair-seq model")

    # for embedding
    parser.add_argument("--llm_truncation_seq_length", default=4096, type=int, required=True, 
                        help="the max seq-length for llm embedding")
    parser.add_argument("--matrix_embedding_exists", action="store_true",
                        help="the seq embedding is or not in advance. default: False")
    parser.add_argument("--emb_dir", default=None, type=str,
                        help="the seq embedding save dir. default: None, not to save")

    # for trained model
    parser.add_argument("--model_path", default=None, type=str, 
                        help="the model dir. default: None")

    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="the dataset name for model building.")
    parser.add_argument("--dataset_type", default=None, type=str, required=True, 
                        help="the dataset type for model building.")
    parser.add_argument("--task_type", default=None, type=str, required=True, 
                        choices=["multi_label", "multi_class", "binary_class", "regression"], 
                        help="the task type for model building.")
    parser.add_argument("--task_level_type", default=None, type=str, required=True, 
                        choices=["seq_level", "token_level"], 
                        help="the task level type for model building.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        choices=["lucabase", "lucapair_homo", "lucapair_heter", "lucatriple_homo", "lucatriple_heter"],
                        help="the model type.")
    parser.add_argument("--input_type", default=None, type=str, required=True, 
                        choices=["seq", "matrix", "vector", "seq-matrix", "seq-vector"], 
                        help="the input type.")
    parser.add_argument("--input_mode", default=None, type=str, required=True, 
                        choices=["single", "pair", "triple"],
                        help="the input mode.")
    parser.add_argument("--time_str", default=None, type=str, required=True, 
                        help="the running time string(yyyymmddHimiss) of model building.")
    parser.add_argument("--step", default=None, type=str, required=True, 
                        help="the training global checkpoint step of model finalization.")

    parser.add_argument("--topk", default=None, type=int, help="the topk labels for multi-class")
    parser.add_argument("--threshold",  default=0.5, type=float, 
                        help="sigmoid threshold for binary-class or multi-label classification, "
                             "None for multi-class classification or regression, default: 0.5.")
    parser.add_argument("--ground_truth_idx", default=None, type=int, 
                        help="the ground truth idx, when the input file contains")
    # for results(csv format, contain header)
    parser.add_argument("--save_path", default=None, type=str, help="the result save path")
    # for print info
    parser.add_argument("--print_per_num", default=10000, type=int, help="per num to print")
    parser.add_argument("--gpu_id", default=None, type=int, help="the used gpu index, -1 for cpu")
    input_args = parser.parse_args()
    return input_args


if __name__ == "__main__":
    run_args = get_args()
    print("-" * 25 + "Run Args" + "-" * 25)
    print(run_args.__dict__)
    print("-" * 50)

    if run_args.input_file is not None:
        input_file_suffix = os.path.basename(run_args.input_file).split(".")[-1]
        if run_args.input_mode in ["pair", "triple"]:
            if input_file_suffix not in ["csv", "tsv"]:
                print("Error! the input file is not in .csv or .tsv format for the pair seqs task.")
                sys.exit(-1)
        else:
            if input_file_suffix in ["fasta", "faa", "fas", "fa"] and run_args.seq_type is None:
                print("Error! input a fasta file, please set arg: --seq_type, value: gene or prot")
                sys.exit(-1)

    # download LLM(LucaOne)
    if not hasattr(run_args, "llm_step"):
        run_args.llm_step = "3800000"
    download_trained_checkpoint_lucavirus(
        llm_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm/"),
        llm_step=run_args.llm_step
    )
    # download trained downstream task models
    '''
    download_trained_checkpoint_downstream_tasks(
        save_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    '''
    if run_args.input_file is not None and os.path.exists(run_args.input_file):
        exists_ids = set()
        exists_res = []
        if os.path.exists(run_args.save_path):
            print("save_path=%s exists." % run_args.save_path)
            if run_args.input_mode == "triple":
                for row in csv_reader(run_args.save_path, header=True, header_filter=True):
                    if len(row) < 8:
                        continue
                    exists_ids.add(row[0] + "_" + row[1] + "_" + row[2])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
            elif run_args.input_mode == "pair":
                for row in csv_reader(run_args.save_path, header=True, header_filter=True):
                    if len(row) < 6:
                        continue
                    exists_ids.add(row[0] + "_" + row[1])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
            else:
                for row in csv_reader(run_args.save_path, header=True, header_filter=True):
                    if len(row) < 4:
                        continue
                    exists_ids.add(row[0])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
        elif not os.path.exists(os.path.dirname(run_args.save_path)):
            os.makedirs(os.path.dirname(run_args.save_path))
        with open(run_args.save_path, "w") as wfp:
            writer = csv.writer(wfp)
            if run_args.input_mode == "triple":
                if run_args.task_type == "multi_class" and run_args.topk is not None and run_args.topk > 1 and run_args.task_level_type == "seq_level":
                    header = [
                        "seq_id_a", "seq_id_b", "seq_id_c",
                        "seq_a", "seq_b", "seq_c",
                        "top1_prob", "top1_label_index", "top1_label",
                        "top%d_probs" % run_args.topk, "top%d_label_indices" % run_args.topk, "top%d_labels" % run_args.topk
                    ]
                else:
                    header = [
                        "seq_id_a", "seq_id_b", "seq_id_c",
                        "seq_a", "seq_b", "seq_c",
                        "prob", "label_index", "label"
                    ]
            elif run_args.input_mode == "pair":
                if run_args.task_type == "multi_class" and run_args.topk is not None and run_args.topk > 1 and run_args.task_level_type == "seq_level":
                    header = [
                        "seq_id_a", "seq_id_b",
                        "seq_a", "seq_b",
                        "top1_prob", "top1_label_index", "top1_label",
                        "top%d_probs" % run_args.topk, "top%d_label_indices" % run_args.topk, "top%d_labels" % run_args.topk
                    ]
                else:
                    header = [
                        "seq_id_a", "seq_id_b",
                        "seq_a", "seq_b",
                        "prob", "label_index", "label"
                    ]
            else:
                if run_args.task_type == "multi_class" \
                        and run_args.topk is not None \
                        and run_args.topk > 1 \
                        and run_args.task_level_type == "seq_level":
                    header = [
                        "seq_id", "seq",
                        "top1_prob", "top1_label_index", "top1_label",
                        "top%d_probs" % run_args.topk, "top%d_label_indices" % run_args.topk, "top%d_labels" % run_args.topk
                    ]
                else:
                    header = [
                        "seq_id", "seq",
                        "prob", "label_index", "label"
                    ]
            if run_args.ground_truth_idx is not None and run_args.ground_truth_idx >= 0:
                header.append("ground_truth")
            writer.writerow(header)
            for item in exists_res:
                writer.writerow(item)
            exists_res = []
            batch_data = []
            batch_ground_truth = []
            had_done = 0

            reader = file_reader(run_args.input_file) if run_args.input_file.endswith(".csv") or run_args.input_file.endswith(".tsv") \
                else fasta_reader(run_args.input_file)
            for row in reader:
                if run_args.input_mode == "triple":
                    if row[0] + "_" + row[1] + "_" + row[2] in exists_ids:
                        continue
                    # seq_id_a, seq_id_b, seq_id_c, seq_type_a, seq_type_b, seq_type_c, seq_a, seq_b, seq_c
                    if not seq_type_is_match_seq(row[3], row[6]):
                        print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                            row[0], row[3], row[6]
                        ))
                        sys.exit(-1)
                    if not seq_type_is_match_seq(row[4], row[7]):
                        print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                            row[1], row[4], row[7]
                        ))
                        sys.exit(-1)
                    if not seq_type_is_match_seq(row[5], row[8]):
                        print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                            row[2], row[5], row[8]
                        ))
                        sys.exit(-1)
                    batch_data.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
                    if run_args.ground_truth_idx is not None and run_args.ground_truth_idx >= 0:
                        batch_ground_truth.append(row[run_args.ground_truth_idx])
                elif run_args.input_mode == "pair":
                    if row[0] + "_" + row[1] in exists_ids:
                        continue
                    # seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
                    if not seq_type_is_match_seq(row[2], row[4]):
                        print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                            row[0], row[2], row[4]
                        ))
                        sys.exit(-1)
                    if not seq_type_is_match_seq(row[3], row[5]):
                        print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                            row[1], row[3], row[5]
                        ))
                        sys.exit(-1)
                    batch_data.append([row[0], row[1], row[2], row[3], row[4], row[5]])
                    if run_args.ground_truth_idx is not None and run_args.ground_truth_idx >= 0:
                        batch_ground_truth.append(row[run_args.ground_truth_idx])
                else:
                    if row[0] in exists_ids:
                        continue
                    if len(row) == 2:
                        if not seq_type_is_match_seq(run_args.seq_type, row[1]):
                            print("Error! the input seq(seq_id=%s) not match its seq_type=%s: %s" % (
                                row[0], run_args.seq_type, row[1]
                            ))
                            sys.exit(-1)
                        batch_data.append([row[0], run_args.seq_type, row[1]])
                    elif len(row) > 2:
                        if not seq_type_is_match_seq(row[1], row[2]):
                            print("Error! the input seq(seq_id=%s) not match its seq_type=%s: %s" % (
                                row[0], row[1], row[2]
                            ))
                            sys.exit(-1)
                        if run_args.ground_truth_idx is not None and run_args.ground_truth_idx >= 0:
                            batch_ground_truth.append(row[run_args.ground_truth_idx])
                        # seq_id, seq_type, seq
                        batch_data.append([row[0], row[1], row[2]])
                    else:
                        continue
                if len(batch_data) % run_args.print_per_num == 0:
                    batch_results = run(
                        batch_data,
                        run_args.llm_truncation_seq_length,
                        run_args.model_path,
                        run_args.dataset_name,
                        run_args.dataset_type,
                        run_args.task_type,
                        run_args.task_level_type,
                        run_args.model_type,
                        run_args.input_type,
                        run_args.input_mode,
                        run_args.time_str,
                        run_args.step,
                        run_args.gpu_id,
                        run_args.threshold,
                        topk=run_args.topk,
                        emb_dir=run_args.emb_dir,
                        matrix_embedding_exists=run_args.matrix_embedding_exists
                    )
                    for item_idx, item in enumerate(batch_results):
                        if run_args.ground_truth_idx is not None and run_args.ground_truth_idx >= 0:
                            item.append(batch_ground_truth[item_idx])
                        writer.writerow(item)
                    wfp.flush()
                    had_done += len(batch_data)
                    print("done %d, had_done: %d" % (len(batch_data), had_done))
                    batch_data = []
                    batch_ground_truth = []
            if len(batch_data) > 0:
                batch_results = run(
                    batch_data,
                    run_args.llm_truncation_seq_length,
                    run_args.model_path,
                    run_args.dataset_name,
                    run_args.dataset_type,
                    run_args.task_type,
                    run_args.task_level_type,
                    run_args.model_type,
                    run_args.input_type,
                    run_args.input_mode,
                    run_args.time_str,
                    run_args.step,
                    run_args.gpu_id,
                    run_args.threshold,
                    topk=run_args.topk,
                    emb_dir=run_args.emb_dir,
                    matrix_embedding_exists=run_args.matrix_embedding_exists
                )
                had_done += len(batch_data)
                for item_idx, item in enumerate(batch_results):
                    if run_args.ground_truth_idx is not None and run_args.ground_truth_idx >= 0:
                        item.append(batch_ground_truth[item_idx])
                    writer.writerow(item)
                wfp.flush()
                batch_data = []
                batch_ground_truth = []
            print("over, had_done: %d" % had_done)
    elif run_args.seq_id is not None \
            and run_args.seq is not None:
        if run_args.seq_type is None:
            print("Please set arg: --seq_type, value: gene or prot")
            sys.exit(-1)
        if not seq_type_is_match_seq(run_args.seq_type, run_args.seq):
            print("Error! the input seq(seq_id=%s) not match its seq_type=%s: %s" % (
                run_args.seq_id, run_args.seq_type, run_args.seq
            ))
            sys.exit(-1)
        data = [[run_args.seq_id, run_args.seq_type, run_args.seq]]
        results = run(
            data,
            run_args.llm_truncation_seq_length,
            run_args.model_path,
            run_args.dataset_name,
            run_args.dataset_type,
            run_args.task_type,
            run_args.task_level_type,
            run_args.model_type,
            run_args.input_type,
            run_args.input_mode,
            run_args.time_str,
            run_args.step,
            run_args.gpu_id,
            run_args.threshold,
            topk=run_args.topk,
            emb_dir=run_args.emb_dir,
            matrix_embedding_exists=run_args.matrix_embedding_exists
        )
        print("Predicted Result:")
        print("seq_id=%s" % run_args.seq_id)
        print("seq=%s" % run_args.seq)
        print("prob=%f" % results[0][2])
        print("label=%s" % results[0][3])
    elif run_args.seq_id_a is not None \
            and run_args.seq_a is not None \
            and run_args.seq_id_b is not None \
            and run_args.seq_b is not None:
        if run_args.seq_type_a is None:
            print("Please set arg: --seq_type_a, value: gene or prot")
            sys.exit(-1)
        if run_args.seq_type_b is None:
            print("Please set arg: --seq_type_b, value: gene or prot")
            sys.exit(-1)
        if not seq_type_is_match_seq(run_args.seq_type_a, run_args.seq_a):
            print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                run_args.seq_id_a, run_args.seq_type_a, run_args.seq_a
            ))
            sys.exit(-1)
        if not seq_type_is_match_seq(run_args.seq_type_b, run_args.seq_b):
            print("Error! the input seq_b(seq_id_b=%s) not match its seq_type_b=%s: %s" % (
                run_args.seq_id_b, run_args.seq_type_b, run_args.seq_b
            ))
            sys.exit(-1)
        data = [[run_args.seq_id_a, run_args.seq_id_b,
                 run_args.seq_type_a, run_args.seq_type_b,
                 run_args.seq_a, run_args.seq_b]]
        results = run(
            data,
            run_args.llm_truncation_seq_length,
            run_args.model_path,
            run_args.dataset_name,
            run_args.dataset_type,
            run_args.task_type,
            run_args.task_level_type,
            run_args.model_type,
            run_args.input_type,
            run_args.input_mode,
            run_args.time_str,
            run_args.step,
            run_args.gpu_id,
            run_args.threshold,
            topk=run_args.topk,
            emb_dir=run_args.emb_dir,
            matrix_embedding_exists=run_args.matrix_embedding_exists
        )
        print("Predicted Result:")
        print("seq_id=%s" % run_args.seq_id)
        print("seq=%s" % run_args.seq)
        print("prob=%f" % results[0][4])
        print("label=%s" % results[0][5])
    elif run_args.seq_id_a is not None \
            and run_args.seq_a is not None \
            and run_args.seq_id_b is not None \
            and run_args.seq_b is not None \
            and run_args.seq_id_c is not None \
            and run_args.seq_c is not None:
        if run_args.seq_type_a is None:
            print("Please set arg: --seq_type_a, value: gene or prot")
            sys.exit(-1)
        if run_args.seq_type_b is None:
            print("Please set arg: --seq_type_b, value: gene or prot")
            sys.exit(-1)
        if run_args.seq_type_c is None:
            print("Please set arg: --seq_type_c, value: gene or prot")
            sys.exit(-1)
        if not seq_type_is_match_seq(run_args.seq_type_a, run_args.seq_a):
            print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (
                run_args.seq_id_a, run_args.seq_type_a, run_args.seq_a
            ))
            sys.exit(-1)
        if not seq_type_is_match_seq(run_args.seq_type_b, run_args.seq_b):
            print("Error! the input seq_b(seq_id_b=%s) not match its seq_type_b=%s: %s" % (
                run_args.seq_id_b, run_args.seq_type_b, run_args.seq_b
            ))
            sys.exit(-1)
        if not seq_type_is_match_seq(run_args.seq_type_c, run_args.seq_c):
            print("Error! the input seq_c(seq_id_c=%s) not match its seq_type_c=%s: %s" % (
                run_args.seq_id_c, run_args.seq_type_c, run_args.seq_c
            ))
            sys.exit(-1)
        data = [[
            run_args.seq_id_a, run_args.seq_id_b, run_args.seq_id_c,
            run_args.seq_type_a, run_args.seq_type_b, run_args.seq_type_c,
            run_args.seq_a, run_args.seq_b, run_args.seq_c
        ]]
        results = run(
            data,
            run_args.llm_truncation_seq_length,
            run_args.model_path,
            run_args.dataset_name,
            run_args.dataset_type,
            run_args.task_type,
            run_args.task_level_type,
            run_args.model_type,
            run_args.input_type,
            run_args.input_mode,
            run_args.time_str,
            run_args.step,
            run_args.gpu_id,
            run_args.threshold,
            topk=run_args.topk,
            emb_dir=run_args.emb_dir,
            matrix_embedding_exists=run_args.matrix_embedding_exists
        )
        print("Predicted Result:")
        print("seq_id_a=%s, seq_id_b=%s" % (run_args.seq_id_a, run_args.seq_id_b))
        print("seq_a=%s" % run_args.seq_a)
        print("seq_b=%s" % run_args.seq_b)
        print("prob=%f" % results[0][6])
        print("label=%s" % results[0][7])
    else:
        raise Exception("input error, usage: --hep")
