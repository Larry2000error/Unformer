import copy
import random
import os
import torch
import json
import ast
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)
import random
random.seed(3407)
import math
from .mre_dataset import shuffle_list_proportion
from .utils import img_augmentations, text_augmentations

class TTTDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False, mismatch=False, mismatch_proportion=0.1, n_views=32) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = '/media/ubuntu/Data/Data/MNRE/mnre/data/'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.write_path = write_path
        self.do_test = do_test
        self.mismatch = mismatch
        self.mismatch_prop = mismatch_proportion
        self.n_views = n_views
        if self.mismatch:
            # old_imgids = copy.deepcopy(self.data_dict['imgids'])
            # self.data_dict['imgids'] = shuffle_list_proportion(self.data_dict['imgids'], self.mismatch_prop)
            # if set(old_imgids) == set(self.data_dict):
            #     print('wrong')
            # 修改dataid而不是imgids,原因:没有读取imgs
            self.data_dict['dataid'], selected_indices = shuffle_list_proportion(self.data_dict['dataid'], self.mismatch_prop)
            self.data_dict['shuffle'] = [0 if i in selected_indices else 1 for i in range(len(self.data_dict['dataid']))]


    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                     self.data_dict['imgids'][idx]

        if self.mismatch:
            shuffle = self.data_dict['shuffle'][idx]
        else:
            shuffle = 1


        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>


        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)

        extend_text_lst = [extend_word_list]
        for _ in range(self.n_views - 1):
            text_aug = random.choice(text_augmentations)
            extend_text_lst.append(text_aug(extend_word_list))

        '''
        BertTokenizer.encode_plus()是HuggingFace，transformers库中的一个方法，用于将文本转换为BERT模型可接受的格式。
        这个方法对输入文本进行编码，并同时处理多种任务，如文本对、填充和截断等。以下是关于encode_plus()的详细解析：
        方法功能:
        encode_plus()将输入文本转换为模型输入所需的格式，包括：
        输入ID（token ids）
        注意力掩码（attention masks）
        Token类型ID（tokentype ids, 用于区分句子对中的两个句子）
        参数:
        encode_plus()
        方法的常用参数包括：
        text: 必需参数，输入的文本或句子。
        text_pair: 可选参数，第二个文本或句子。用于处理文本对的任务（如问答任务）。
        add_special_tokens: 可选参数，是否在编码中添加特殊标记（如[CLS]和[SEP]）。默认值是True。
        max_length: 可选参数，输入的最大长度。如果超出此长度，则会进行截断；如果短于此长度，则会进行填充。
        padding: 可选参数，指定是否对输入进行填充。可以设置为True（填充到max_length）、False（不填充）、longest（填充到批次中的最长序列）。
        truncation: 可选参数，指定是否对输入进行截断。可以设置为True（截断到max_length）、False（不截断）。
        return_tensors: 可选参数，指定返回的张量类型。可以设置为'pt'（PyTorch张量）或'tf'（TensorFlow张量）。
        返回值:
        encode_plus()返回一个字典，包含模型所需的所有输入信息，通常包括：
        input_ids: 编码后的token ids。
        token_type_ids: 对于文本对任务，区分两个句子的token类型ID。
        attention_mask: 注意力掩码，指示模型应该关注哪些token（非填充部分
        '''
        input_lst, token_type_lst, attention_masks = [], [], []
        for text in extend_text_lst:
            encode_dict = self.tokenizer.encode_plus(text=text, max_length=self.max_seq, truncation=True,
                                                     padding='max_length')
            input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                encode_dict['attention_mask']
            input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(
                token_type_ids), torch.tensor(attention_mask)
            input_lst.append(input_ids), token_type_lst.append(token_type_ids), attention_masks.append(attention_mask)

        # encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
        #                                          padding='max_length')
        # input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
        #                                             encode_dict['attention_mask']
        # input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
        #     attention_mask)

        re_label = self.re_dict[relation]  # label to id

        '''
        CLIPProcessor是HuggingFace transformers库中的一个处理器，用于预处理图像和文本，以适应CLIP模型的输入格式。当你使用
        CLIPProcessor对图像进行处理时，例如通过CLIPProcessor(images=aux_img, return_tensors='pt')，它返回一个包含处理后图像数据的字典。
        这个字典通常包括以下内容：
        pixel_values: 处理后的图像数据，以PyTorch张量的形式返回（如果return_tensors = 'pt'）
        '''

        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                orig_aux_imgs = []
                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    orig_aux_imgs.append(aux_img)
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)


                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

                    orig_rcnn_imgs = []
                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        orig_rcnn_imgs.append(rcnn_img)
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3

                    aux_imgs_lst, rcnn_imgs_lst = [aux_imgs], [rcnn_imgs]
                    for _ in range(self.n_views - 1):
                        aug_aux_imgs, aug_rcnn_imgs = [], []
                        img_aug = random.choice(img_augmentations)
                        for aux_img in orig_aux_imgs:
                            aug_aux_img = img_aug(aux_img, 1)
                            aug_aux_img = self.aux_processor(images=aug_aux_img, return_tensors='pt')['pixel_values'].squeeze()
                            aug_aux_imgs.append(aug_aux_img)
                        # padding
                        for i in range(3 - len(aug_aux_imgs)):
                            aug_aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                        assert len(aug_aux_imgs) == 3

                        for rcnn_img in orig_rcnn_imgs:
                            aug_rcnn_img = img_aug(rcnn_img, 1)
                            aug_rcnn_img = self.rcnn_processor(images=aug_rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                            aug_rcnn_imgs.append(aug_rcnn_img)
                        # padding
                        for i in range(3 - len(aug_rcnn_imgs)):
                            aug_rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                        assert len(aug_rcnn_imgs) == 3

                        aux_imgs_lst.append(torch.stack(aug_aux_imgs, dim=0))
                        rcnn_imgs_lst.append(torch.stack(aug_rcnn_imgs, dim=0))

                    # 图像 + 文本增强
                    # return torch.stack(input_lst, dim=0), torch.stack(token_type_lst, dim=0), torch.stack(attention_masks, dim=0), torch.tensor(re_label).repeat(self.n_views, 1), image.repeat((self.n_views, 1, 1, 1)), \
                    #        torch.stack(aux_imgs_lst, dim=0), torch.stack(rcnn_imgs_lst, dim=0), torch.tensor(shuffle).repeat(self.n_views, 1)
                    # 文本增强
                    # return torch.stack(input_lst, dim=0), torch.stack(token_type_lst, dim=0), torch.stack(attention_masks, dim=0), torch.tensor(re_label).repeat(self.n_views, 1), image.repeat(
                    #     (self.n_views, 1, 1, 1)), aux_imgs.repeat((self.n_views, 1, 1, 1, 1)), rcnn_imgs.repeat((self.n_views, 1, 1, 1, 1)), torch.tensor(shuffle).repeat(self.n_views, 1)
                    return input_ids.repeat(self.n_views, 1, 1), token_type_ids.repeat(self.n_views, 1, 1), attention_mask.repeat(self.n_views, 1, 1), torch.tensor(re_label).repeat(self.n_views, 1), image.repeat(
                        (self.n_views, 1, 1, 1)), torch.stack(aux_imgs_lst, dim=0), torch.stack(rcnn_imgs_lst, dim=0), torch.tensor(shuffle).repeat(self.n_views, 1)
                    # return input_ids, token_type_ids, attention_mask, torch.tensor(
                    #         re_label), image, aux_imgs, rcnn_imgs, torch.tensor(shuffle)
