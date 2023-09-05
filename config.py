import torch
class Config(object):
    def __init__(self):
    
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 50
        self.use_en = True 
        #默认使用英文tokenizer， 使用中文时，值为False
        self.rnn_hidden = 500
        # self.bert_embedding = 1920
        self.bert_embedding = 768
        # self.bert_embedding = 896
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 5e-5
        self.lr_decay = 0.00005
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 100
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':

    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)

