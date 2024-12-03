import copy
import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)
import random
random.seed(1)
import math

class MREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        aux_imgs = None
        aux_path = self.data_path[mode + "_auximgs"]
        aux_imgs = torch.load(aux_path)
        rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])
        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    # relation and corresponding train samples
    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class MREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False, mismatch=False, mismatch_proportion=0.1) -> None:
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
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

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

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
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

                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid, torch.tensor(shuffle)
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, torch.tensor(shuffle)

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, torch.tensor(shuffle)

        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), torch.tensor(shuffle)

class MRETextDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False, mismatch=False, mismatch_proportion=0.1) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.write_path = write_path
        self.do_test = do_test


    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx],

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
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = self.re_dict[relation]  # label to id

        '''
        CLIPProcessor是HuggingFace transformers库中的一个处理器，用于预处理图像和文本，以适应CLIP模型的输入格式。当你使用
        CLIPProcessor对图像进行处理时，例如通过CLIPProcessor(images=aux_img, return_tensors='pt')，它返回一个包含处理后图像数据的字典。
        这个字典通常包括以下内容：
        pixel_values: 处理后的图像数据，以PyTorch张量的形式返回（如果return_tensors = 'pt'）
        '''
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)


def shuffle_list_proportion(lst, proportion):
    # 计算需要打乱的元素数量
    num_items = len(lst)
    num_shuffle = math.ceil(num_items * proportion)

    # 选择需要打乱的元素
    indices = list(range(num_items))
    selected_indices = random.sample(indices, num_shuffle)

    # 提取选中的元素
    selected_items = [lst[i] for i in selected_indices]

    # 打乱选中的元素
    shuffled_items = selected_items[:]
    while True:
        random.shuffle(shuffled_items)
        # 确保打乱后的元素不在原位置
        if all(original != new for original, new in zip(selected_items, shuffled_items)):
            break

    # 更新列表
    for i, new_value in zip(selected_indices, shuffled_items):
        lst[i] = new_value

    return lst, selected_indices,




