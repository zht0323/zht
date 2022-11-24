import numpy as np
from tqdm import tqdm
from config import Config
import pandas as pd
from transformers import BertTokenizer
from textrank4zh import TextRank4Sentence
import re
from snownlp import SnowNLP

def MinMaxsclar(num):
    '''归一化'''
    lst = []
    Max = max(num)
    Min = min(num)
    for i in num:
        lst.append(i / sum(num))
    return lst

def num_create(long_text):
    '''计算出这个长句子的分局有多少个'''
    text_list = long_text[0].split('。')
    num = len(text_list)
    return num

def weight_create(num,long_text):
    '''生成短句子和其对应的权重值'''
    tr4s = TextRank4Sentence()
    tr4s.analyze(long_text[0], lower=True, source='all_filters')
    keysentences = tr4s.get_key_sentences(num=num // 2, sentence_min_len=1)

    sentence = []
    weight = []
    list_sentence = []
    list_weight = []
    for num in keysentences:
        list_sentence.append(num['sentence'])
        list_weight.append(num['weight'])
    a = tuple(list_sentence)
    b = tuple(list_weight)
    sentence.append(a)
    list_sentence = []
    weight.append(b)
    list_weight = []
    # 归一化一下
    weight = MinMaxsclar(weight[0])

    return sentence , weight

def load_data(sentence:str):
    """
    读取数据
    """
    long_text = sentence
    mini_seq_num = num_create(long_text)
    sentence , weight = weight_create(mini_seq_num,long_text)
    lines = list(zip(sentence[0], weight))
    return lines , mini_seq_num

class InputExample():
    def __init__(self,sentence,weight):
        self.sentence = sentence
        self.weight = weight

def create_example(lines):

    examples = []
    for one in lines:
        sentence = one[0]
        weight = int(one[1])
        examples.append(InputExample(sentence=sentence, weight=weight))
    return examples

def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )

class DataIterator:
    """
    数据迭代器
    """
    def __init__(self, batch_size, sentence, tokenizer, use_bert=False, seq_length=50, is_test=False):
        self.sentence = sentence
        self.data,self.mini_seq_num= load_data(sentence)    #('第一节财务公债券有价证券投资对金融机构的股权投资',0.008331081059317274) * mini_seq_num个
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)     # 1个大长句子 中 小句子的个数
        self.all_tags = []
        self.idx = 0                          # 指针,指向小句子个数
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        self.tokenizer = tokenizer

    def convert_single_example(self, mini_seq):
        '''生成可以用的字向量 , mini_seq这个参数是每个大长句子拆成  mini_seq_num 多个小句子, mini_seq是指针, 指向 range(0 , mini_seq_num)'''

        # 第 mini_seq 个小句子的计算
        sentence = self.data[mini_seq][0]
        weight = self.data[mini_seq][1]

        q_tokens = []
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # 得到text_a的 token  按字去切
        for word in sentence:
            token = self.tokenizer.tokenize(word)
            q_tokens.extend(token)

        # 在q_tokens的基础上加上了"[CLS]",存入ntokens,用做后续的token2id
        for token in q_tokens:
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(1)

        # 按照你设定的最大长度去截取
        ntokens = ntokens[:self.seq_length - 1]
        segment_ids = segment_ids[:self.seq_length - 1]

        #加入 [SEP]
        ntokens.append("[SEP]")
        segment_ids.append(1)

        #token2id
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length

        return input_ids, input_mask, segment_ids, weight

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        '''固定和下面的__next__组合成用法,用来说明这个类是个迭代器,
        在数据传入进来后,经过__init__后,进入__next__,直到你传入的数据按照迭代器的规则运行完'''
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        weight_list = []
        num_tags = 0

        # 每次返回batch_size个数据
        while num_tags < self.batch_size:
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, weight = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            weight_list.append(weight)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break
        while len(input_ids_list) < self.batch_size:
            input_ids_list.append(input_ids_list[0])
            input_mask_list.append(input_mask_list[0])
            segment_ids_list.append(segment_ids_list[0])
            weight_list.append(weight_list[0])
        print('okay')
        return input_ids_list, input_mask_list, segment_ids_list, weight_list, self.seq_length


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    # train_iter = DataIterator(config.batch_size,
    #                           sentence=sentence, use_bert=config.use_bert,
    #                           tokenizer=tokenizer, seq_length=config.max_seq_len)
    print('okay')

    # dev_iter = DataIterator(config.batch_size,
    #                         data_file=config.dev_file, use_bert=config.use_bert,
    #                         seq_length=config.max_seq_len, is_test=True,
    #                         tokenizer=tokenizer)



