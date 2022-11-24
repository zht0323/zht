import torch.nn as nn
from config import Config
import torch
import numpy as np
from transformers import BertTokenizer


config = Config()
if config.pretrainning_model == 'nezha':
    from NEZHA.model_nezha import BertPreTrainedModel, NEZHAModel
elif config.pretrainning_model == 'albert':
    from transformers import AlbertModel, BertPreTrainedModel
else:
    # bert,roberta
    from transformers import BertPreTrainedModel, BertModel
class BertForCLS(BertPreTrainedModel):
    def __init__(self, config, params):
        #config是调用的nezha的config文件,params是调用的自己写的config
        super().__init__(config)   #继承config类，并且在此基础上添加下面参数
        self.params = params
        self.config = config
        # 预训练模型
        if params.pretrainning_model == 'nezha':  # batch_size, max_len, 768
            self.bert = NEZHAModel(config)
        elif params.pretrainning_model == 'albert':
            self.bert = AlbertModel(config)
        else:
            # self.bert = RobertaModel(config)
            self.bert = BertModel(config)

        #  动态权重组件
        self.classifier = nn.Linear(config.hidden_size, 1)  # for dym's dense
        self.dym_pool = nn.Linear(params.embed_dense, 1)  # for dym's dense
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, params.embed_dense),
                                         nn.ReLU(True))  # 动态最后的维度
        self.dense_emb_size = nn.Sequential(nn.Linear(config.hidden_size, params.embed_dense),
                                         nn.ReLU(True))  # 降维
        #这里用nn.Parameter是为了将torch.ones生成的矩阵变成可训练的参数矩阵,传入模型
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)
        # self.pool_weight = nn.Parameter(torch.ones((params.batch_size, 1, 1, 1)),
        #                                 requires_grad=True)

        # 下游结构组件
        if params.model_type == 'bilstm':
            num_layers = params.rnn_num
            lstm_num = int(self.params.embed_dense / 2)
            self.lstm = nn.LSTM(self.params.embed_dense, lstm_num,
                                num_layers, batch_first=True,  # 第一维度是否为batch_size
                                bidirectional=True)  # 双向
        elif params.model_type == 'bigru':
            num_layers = params.rnn_num
            lstm_num = int(self.params.embed_dense / 2)
            self.lstm = nn.GRU(self.params.embed_dense, lstm_num,
                               num_layers, batch_first=True,  # 第一维度是否为batch_size
                               bidirectional=True)  # 双向
        # 全连接分类组件
        self.cls = nn.Linear(params.embed_dense, params.cls_num)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if params.pretrainning_model == 'nezha':
            self.apply(self.init_bert_weights)
        else:
            self.init_weights()
        self.reset_params()

    def reset_params(self):
        '''通过Xavier初始化可学习的参数矩阵dym_weight,使得输入和输出的方差保持不变'''
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):

        layer_logits = []
        # print(len(outputs))
        all_encoder_layers = outputs[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))     #整合全句子去自己做了个cls的映射
        # print('layer_logits::::原始:::', len(layer_logits))

        # layer_logits这个列表的每一个[batch_size , 512 , 1]按照列去concat,最后成了[batch_size ,512 , 12]
        layer_logits = torch.cat(layer_logits, 2)
        # print('layer_logits:::::::', layer_logits.shape)
        #这里维度没有变化,就相当于做了个加权的标准化
        layer_dist = torch.nn.functional.softmax(layer_logits)
        # print('layer_dist:::::::',layer_dist.shape)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)
        word_embed = self.dense_final(pooled_output)
        dym_layer = word_embed
        return dym_layer

    def get_weight_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def clsCreate(self,data:list,DataIterator):
        '''针对每一句话生成一群batch , 综合权重考虑每一句话 , 生成一个cls'''
        inputs = []  # 用来记录每一个句子的最终的 CLS向量

        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                                  do_lower_case=True,
                                                  never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
        for sentence in data:
            train_iter = DataIterator(config.batch_size,
                                      sentence=sentence, use_bert=config.use_bert,
                                      tokenizer=tokenizer, seq_length=config.max_seq_len)
            input_ids_list, _,_, weight_list,_ = train_iter
            # 这边摘出来的是最后一个层的encoder结果
            encoded_layers, ori_pooled_output = self.bert(
                input_ids_list,
                attention_mask=None,
                token_type_ids=None,
                output_all_encoded_layers=False
            )





    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                cls_label=None):
        '''input_ids是将每个字按照自检对应的id去转换成数字，去传入到forword里面的'''
        # 预训练模型
        encoded_layers, ori_pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_all_encoded_layers=False
        )
        sequence_output = self.dense_final(encoded_layers)  #

        # 下游任务
        if self.params.model_type == 'bilstm' or self.params.model_type == 'bigru':
            # self.lstm的输出是两个值,一个是output , 另一个是(h , c) ,这里我们需要的是output , 所以 提取的是输出的 [0]
            sequence_output = self.lstm(sequence_output)[0]
            #这里由于数采用的双向lstm结构,hidden_size设置的是 embed_dense / 2,所以出来的时候还是和 seq_len 一样

        pooled_output = ori_pooled_output
        pooled_output = self.dense_emb_size(pooled_output)

        # 分类
        # drop(0.1)
        cls_output = self.dropout(pooled_output)
        classifier_logits = self.cls(cls_output)  # [bacth_size , cls_num]

        outputs = classifier_logits, encoded_layers  #

        return outputs
if __name__ == '__main__':
