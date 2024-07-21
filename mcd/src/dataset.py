import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import json
from nltk.tokenize import wordpunct_tokenize
import string


class MCDDataset(Dataset):
    def __init__(self, base_path, path_vid):
        self.video_feas = h5py.File(
            os.path.join(base_path, "video_feature_clip_chinese_vit_h.h5"), "r"
        )["video_feas"]
        self.title_feas = h5py.File(
            os.path.join(base_path, "title_feature_clip.h5"), "r"
        )
        self.comment_feas = h5py.File(
            os.path.join(base_path, "comment_feature_clip.h5"), "r"
        )
        self.author_feas = h5py.File(
            os.path.join(base_path, "author_feature_clip.h5"), "r"
        )
        self.asr_feas = h5py.File(os.path.join(base_path, "asr_feature_clip.h5"), "r")
        self.data = []
        self.senticNet = pickle.load(
            open("dataset/senticnet_word.pkl", "rb")
        )
        self.comment_trans = pickle.load(
            open("dataset/translate/translate_comments.pkl", "rb")
        )
        for k in self.comment_trans.keys():
            self.comment_trans[k] = [
                re.sub(
                    "[0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/\\\[\]'\"\u4e00-\u9fa5]",
                    "",
                    i,
                )
                .strip()
                .lower()
                for i in self.comment_trans[k]
            ]
        with open("dataset/metadata_clean.json", "r", encoding="utf-8-sig") as f:
            self.data_complete = f.readlines() 
        with open(os.path.join(base_path, "data-split/", path_vid), "r") as fr:
            self.vid = fr.readlines()
        self.vid = [i.replace("\n", "") for i in self.vid]
        self.data = []
        for i in self.data_complete:
            i = json.loads(i)
            if i["video_id"] in self.vid:
                self.data.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = str(item["video_id"])
        # label
        label = torch.tensor(item["controversy"])
        # video
        video_fea = self.video_feas[vid][:]
        video_fea = torch.FloatTensor(video_fea)
        # title
        title_fea = self.title_feas[vid][:]
        title_fea = torch.FloatTensor(title_fea)
        # comments
        if vid in self.comment_feas.keys():
            comments_fea = self.comment_feas[vid][:]
            comments_fea = torch.FloatTensor(comments_fea)
            comment_lens = len(comments_fea[:40])
        else:
            comments_fea = torch.FloatTensor(1024)
            comment_lens = 1
        # author
        author_fea = self.author_feas[vid][:]
        author_fea = torch.FloatTensor(author_fea)
        # asr
        asr_fea = self.asr_feas[vid][:]
        asr_fea = torch.FloatTensor(asr_fea)
        if vid in self.comment_trans.keys():
            sen_adj = sentic_adj_matrix(self.comment_trans[vid], self.senticNet, 40)
        else:
            sen_adj = sentic_adj_matrix("", self.senticNet, 40)
        return {
            "vid": vid,
            "label": label,
            "video_fea": video_fea,
            "title_fea": title_fea,
            "comments_fea": comments_fea,
            "comment_lens": comment_lens,
            "author_fea": author_fea,
            "asr_fea": asr_fea,
            "sen_adj": sen_adj,
        }


def sentic_adj_matrix(sen_list, senticNet, length):
    tokenize = lambda s: wordpunct_tokenize(
        re.sub("[%s]" % re.escape(string.punctuation), " ", s)
    )  
    seq_len = min(len(sen_list), 40)
    matrix = np.zeros((length, length)).astype("float32")
    sentic_list = [0] * seq_len
    for i in range(seq_len):
        sentic_list[i] = 0
        for j in tokenize(sen_list[i]):
            if j not in senticNet:
                continue
            sentic_list[i] += senticNet[j]
    for i in range(seq_len):
        for j in range(i, seq_len):
            if (
                sentic_list[i] == 0
                or sentic_list[j] == 0
                or sentic_list[i] == sentic_list[j]
            ):
                continue
            sentic = abs(float(sentic_list[i] - sentic_list[j]))
            matrix[i][j] = sentic
            matrix[j][i] = sentic
    matrix = torch.FloatTensor(matrix)
    return matrix


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        if len(video.shape) == 1:
            video = torch.zeros((1, 1024))
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat(
                (
                    video,
                    torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float),
                ),
                dim=0,
            )
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def MCD_collate_fn(batch):
    num_comments = 40
    num_frames = 80
    vids = [int(item["vid"]) for item in batch]
    labels = [item["label"] for item in batch]
    video_feas = [item["video_fea"] for item in batch]
    video_feas, video_masks = pad_frame_sequence(num_frames, video_feas)
    asr_feas = [item["asr_fea"] for item in batch]
    title_feas = [item["title_fea"] for item in batch]
    comment_lens = [len(item["comments_fea"][:40]) for item in batch]
    comment_feas = [item["comments_fea"] for item in batch]
    comment_feas, comment_masks = pad_frame_sequence(num_comments, comment_feas)
    author_feas = [item["author_fea"] for item in batch]
    sen_adj = [item["sen_adj"] for item in batch]

    return {
        "vid": torch.tensor(vids),
        "label": torch.stack(labels),
        "video_feas": video_feas,
        "video_masks": video_masks,
        "title_feas": torch.stack(title_feas),
        "comment_feas": comment_feas,
        "comment_lens": torch.IntTensor(comment_lens),
        "author_feas": torch.stack(author_feas),
        "asr_feas": torch.stack(asr_feas),
        "sen_adj": torch.stack(sen_adj),
    }
