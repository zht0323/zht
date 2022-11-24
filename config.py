'''
集成所有用到的参数的一个类
'''
class Config():
    def __init__(self):
        #文件参数
        self.data_file = './data/train.csv'
        # self.dev_file = './data/Test_DataSet.csv'


        #模型参数
        self.batch_size = 30
        self.max_seq_len = 50


        self.use_bert = True

        # 模型选择  NEZHA
        model = 'E:/ZJQ/hugging_cls/nezha-cn-base/'  # 中文nezha-base

        self.params.use_origin_bert == 'dym'

        self.model_path = model
        self.bert_file = model + 'pytorch_model.bin'
        self.bert_config_file = model + 'config.json'
        self.vocab_file = model + 'vocab.txt'