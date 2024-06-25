import os
import argparse
import json
import ruamel_yaml as yaml

from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from data.utils import pre_question

class vqa_dataset(Dataset):
    def __init__(self, transform, image_list, question_list, answer_list, split='train'):
        if split not in ['train', 'test']:
            raise Exception('split must be "train" or "test"!')
        self.split = split
        self.transform = transform
        self.image_list = image_list
        self.question_list = question_list
        self.answer_list = answer_list

    def __len__(self):
        return len(self.answer_list)

    def __getitem__(self, index):
        image = np.array(self.image_list[index])
        image = self.transform(image)

        question = pre_question(self.question_list[index])

        if (self.split == 'train'):
            answers = [pre_question(self.answer_list[index])]
            weights = [0.2]
            return image, question, answers, weights
        else:
            answers = pre_question(self.answer_list[index])
            return image, question, answers
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n