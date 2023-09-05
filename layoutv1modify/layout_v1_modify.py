import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '..')))
"""
采用layout+lstmcrf,不从采用GAT,和layout+gat+lstmcrf使用GAT的 效果进行对比

"""


from operator import itemgetter
from pickle import NONE
import re
from select import select
from sys import api_version
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import scipy.sparse as sp

import random
import json
import time
import numpy as np
from torchvision.transforms import ToTensor
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer


#导入lstmcrf模块
from lstm_crf import BERT_LSTM_CRF
import torch.optim as optim
from config import Config


import numpy as np
from seqeval.metrics import (
      classification_report,
      f1_score,
      precision_score,
      recall_score,
  )
from os import listdir
#使得导入的模型生效
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)



config = Config()
save_model_path = "/data/experiments/1data/savedmodel/v1modify/"
# list all training image file names
#listdir返回的结果本身就是一个 list，所以这里用列表生成表达式就是多次一举
# list all training image file names
image_files_train = [f for f in listdir('/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/dataset_normal/training_data/images')]
# list all test image file names
image_files_test = [f for f in listdir('/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/dataset_normal/testing_data/images')]


#导入标签
from custon_datasetwithpad import idx2label,label2idx

# from label import idx2label,label2idx
def normalize_box(box, width, height):
     return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

# resize corresponding bounding boxes (annotations)
# Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
def resize_and_align_bounding_box(bbox, original_image, target_size):
  x_, y_ = original_image.size

  x_scale = target_size / x_ 
  y_scale = target_size / y_
  
  origLeft, origTop, origRight, origBottom = tuple(bbox)
  
  x = int(np.round(origLeft * x_scale))
  y = int(np.round(origTop * y_scale))
  xmax = int(np.round(origRight * x_scale))
  ymax = int(np.round(origBottom * y_scale)) 
  
  return [x-0.5, y-0.5, xmax+0.5, ymax+0.5]


class FUNSDDataset(Dataset):
    """LayoutLM dataset with visual features."""

    def __init__(self, image_file_names, tokenizer, max_length, target_size, train=True, predict = False):
        self.image_file_names = image_file_names
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.target_size = target_size
        self.pad_token_box = [0, 0, 0, 0]
        self.train = train
        self.predict = predict

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):

        # first, take an image,得到一个图像的文件名
        item = self.image_file_names[idx]

        if self.train:
          base_path = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/dataset_normal/training_data"
        elif self.predict:
          base_path = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/layoutlm_data"
        else:
          base_path = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/dataset_normal/testing_data"
        
        original_image = Image.open(base_path + "/images/" + item).convert("RGB")
        # resize to target size (to be provided to the pre-trained backbone)
        resized_image = original_image.resize((self.target_size, self.target_size))
        
        # first, read in annotations at word-level (words, bounding boxes, labels)
        with open(base_path + '/annotations/' + item[:-4] + '.json') as f:
          data = json.load(f)
        words = []
        unnormalized_word_boxes = []
        word_labels = []
        for annotation in data:
          # get label
          label = annotation['label']
          # get words,annotation['words']是一个列表，里面可能有多个字典，每个字典中有'text'项和'box项'
          
          if annotation['text'] == '':
            continue
          words.append(annotation['text'])
          unnormalized_word_boxes.append(annotation['position'])

          word_labels.append(label)

        width, height = original_image.size
        #经过normalize_box后四个坐标中第2个和第4个没变，是因为图像的height有的为1000，这样分子分母都是1000，就抵消了
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        for box in normalized_word_boxes:

          for i in range(len(box)):
            if box[i] > 1000:
              box[i] = 1000
            if box[i] < 0:
              box[i] = 0
        
        assert len(words) == len(normalized_word_boxes)

        # next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
        #token_boxes中存放的是 经过 normalize_box()函数处理过的坐标
        token_boxes = []

        #unnormalized_token_boxes存放的是原始的未经处理的坐标，后面的 esize_and_align_bounding_box()函数会用 
        unnormalized_token_boxes = []
        token_labels = []


        for word, unnormalized_box, box, label in zip(words, unnormalized_word_boxes, normalized_word_boxes, word_labels):

            #对于ocr得到的结果的json文件中, 除了Date:这种，其他的一般都已经是单个单词了，所以这里经过self.tokenizer.tokenize得到的结果大多数 都还是原来的token，
            #只有Date:这种 被分为 ["Date",":"]2个token
            word_tokens = self.tokenizer.tokenize(word)
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
            # label first token as B-label (beginning), label all remaining tokens as I-label (inside)
            for i in range(len(word_tokens)):
                  if i == 0:
                    token_labels.extend(['B-' + label])
                  else:
                    token_labels.extend(['I-' + label])
            
            # if label == "0":


            #   for i in range(len(word_tokens)):

            #     token_labels.extend([label])

            # else:
            
            #   for i in range(len(word_tokens)):
            #     if i == 0:
            #       token_labels.extend(['B-' + label])
            #     else:
            #       token_labels.extend(['I-' + label])

     
        # Truncation of token_boxes + token_labels
        special_tokens_count = 2 
        if len(token_boxes) > self.max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
            token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]
        
        # add bounding boxes and labels of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
        # token_labels = [-100] + token_labels + [-100]
        token_labels = [0] + token_labels + [0]
        
        #新添加的语句，可能是因为transformers的版本问题。因为设置了paddig="max_length"，所以input_ids,attention_mask,token_type_ids的长度都是512
        encoding = self.tokenizer(text=' '.join(words),
                  padding="max_length",
                  max_length=512,
                  truncation=True)
                  

        #encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        #注释掉原有的语句
        # Padding of token_boxes up the bounding boxes to the sequence length.

        #获取实际长度，为了下面添加token_boxes和token_labels属性，并使得其长度也为512
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = self.max_seq_length - len(input_ids)
        token_boxes += [self.pad_token_box] * padding_length
        unnormalized_token_boxes += [self.pad_token_box] * padding_length
        # token_labels += [-100] * padding_length
        token_labels += [0] * padding_length

        #添加token_boxes和token_labels属性
        encoding['bbox'] = token_boxes
        encoding['labels'] = token_labels

      
        assert len(encoding['input_ids']) == self.max_seq_length  #[512]
        
        assert len(encoding['attention_mask']) == self.max_seq_length  #[]
        assert len(encoding['token_type_ids']) == self.max_seq_length 
        assert len(encoding['bbox']) == self.max_seq_length    #[512,4]
        assert len(encoding['labels']) == self.max_seq_length  #[512]

        encoding['resized_image'] = ToTensor()(resized_image)
        # rescale and align the bounding boxes to match the resized image size (typically 224x224) 
        #添加resized_and_aligned_bounding_boxes属性,长度也为512，因为unnormalized_token_boxes已经经过了padding
      
        encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, self.target_size) 
                                                          for bbox in unnormalized_token_boxes]  #[512,4]

        encoding['unnormalized_token_boxes'] = unnormalized_token_boxes
        
        # finally, convert everything to PyTorch tensors 
        for k,v in encoding.items():
            #把label字符串转化为labletoindex中对应的索引
            if k == 'labels':
              label_indices = []
              # convert labels from string to indices
              for label in encoding[k]:
                # if label != -100:
                if label != 0:
                  label_indices.append(label2idx[label])
                else:
                  label_indices.append(label)
              encoding[k] = label_indices
            #把encoding中的其他属性也 由list转化为 tensor
            encoding[k] = torch.as_tensor(encoding[k])

        return encoding


#定义模型
#So if we want to define a model that includes LayoutLM + the visual embeddings, it would look like this:

import torch.nn as nn
from transformers.models.layoutlm import LayoutLMModel, LayoutLMConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torchvision
from torchvision.ops import RoIAlign

from transformers import AdamW
from tqdm.notebook import tqdm



class LayoutLMForTokenClassification(nn.Module):
    def __init__(self, output_size=(3,3), 
                 spatial_scale=14/224, 
                 sampling_ratio=2
        ): 
        super().__init__()

        #定义bert模型-------------
        self.model_name = 'bert-base-uncased'

        self.config = Config()

        # a.通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        # b. 导入配置文件
        self.model_config = BertConfig.from_pretrained(self.model_name)

        # 通过配置和路径导入模型
        self.bert_model = BertModel.from_pretrained(self.model_name, config = self.model_config)

        
        # LayoutLM base model + token classifier
        self.num_labels = len(label2idx)  # 8
        self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=self.num_labels)
        self.dropout = nn.Dropout(self.layoutlm.config.hidden_dropout_prob)

        self.dropout1 = nn.Dropout(0.55)
        self.classifier = nn.Linear(self.layoutlm.config.hidden_size, self.num_labels)

        # backbone + roi-align + projection layer
        model = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-3]))
        self.roi_align = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
        self.projection = nn.Linear(in_features=1024*3*3, out_features=self.layoutlm.config.hidden_size)
      
      
    def forward(
        self,
        input_ids,  #[batchsize,max_length]
        bbox,
        attention_mask,
        token_type_ids,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        resized_images=None, # shape (N, C, H, W), with H = W = 224
        resized_and_aligned_bounding_boxes=None, # [batchsize,max_length,4],single torch tensor that also contains the batch index for every bbox at image size 224
       
        boundingbox_text_len=None,
        sentence_boxes=None,

        NodesLabelToIndex=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        """
        return_dict = return_dict if return_dict is not None else self.layoutlm.config.use_return_dict

        # first, forward pass on LayoutLM
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) # odict_keys(['last_hidden_state', 'pooler_output'])

        
      
        """
        layoutLM和bert的使用几乎完全一样，只不过在bert的输入特征的基础上，又增加了2D位置特征和图像特征，最后的的输出也是包含2部分，
        CLS('pooler_output')和 每个token('last_hidden_state':文本+2D位置+图像)的输出
        
        """
        
        sequence_output = outputs[0] # [batchsize,max_length,hiddensize]->[10,512,768]

        pooler_output = outputs[1] # [batchsize,hiddensize]

        # next, send resized images of shape (batch_size, 3, 224, 224) through backbone to get feature maps of images 
        #接下来的代码使用 resnet101和 RoIAlign获得图像的特征 
        # shape (batch_size, 1024, 14, 14)
        feature_maps = self.backbone(resized_images)  #[batchsize,1024,14,14]
        
        # next, use roi align to get feature maps of individual (resized and aligned) bounding boxes
        # shape (batch_size*seq_len, 1024, 3, 3)
        resized_bounding_boxes_list = []
        for i in resized_and_aligned_bounding_boxes:
          resized_bounding_boxes_list.append(i.float().to(config.device))
                
        feat_maps_bboxes = self.roi_align(input=feature_maps, 
                                        # we pass in a list of tensors
                                        # We have also added -0.5 for the first two coordinates and +0.5 for the last two coordinates,
                                        # see https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
                                        rois=resized_bounding_boxes_list
                           )  # [batchsize,max_length,9216]
      
        # next, reshape  + project to same dimension as LayoutLM. 

        #获得图像的特征后，因为要 和 layoutLM获得的特征相加，所以 要先进行view，进行对其操作，统一为:[batchsize,maxNumNodegth,hiddensize] 
        batch_size = input_ids.shape[0] #[batchsize]
        seq_len = input_ids.shape[1]  #[max_length]
        feat_maps_bboxes = feat_maps_bboxes.view(batch_size, seq_len, -1) # Shape (batch_size, seq_len, 1024*3*3)
        projected_feat_maps_bboxes = self.projection(feat_maps_bboxes) # Shape (batch_size, seq_len, hidden_size)



        # add those to the sequence_output - shape (batch_size, seq_len, hidden_size)
        sequence_output += projected_feat_maps_bboxes  

        return  sequence_output


#将各层统一起来

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        config = Config()
        num_labels = len(label2idx) 
        self.dropout = nn.Dropout(0.5)
        self.layoutlm = LayoutLMForTokenClassification()
        self.lstm_crf = BERT_LSTM_CRF(num_labels, config.bert_embedding, config.rnn_hidden,
 config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1,
 use_cuda=config.use_cuda)

        self.linear = nn.Linear(768,len(label2idx)+2)

        self.bn = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
    def forward(self,input_ids, bbox, attention_mask, token_type_ids, 
        labels, resized_images, resized_and_aligned_bounding_boxes,
        trainer=True):
        h = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
        labels=labels, resized_images=resized_images, 
        resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes
        )
        h = F.relu(h)
        # h = self.bn(h)
        h = self.dropout(h)
        h = self.lstm_crf(h, attention_mask)
        # h = self.linear(h)

        if trainer:
          lstmcrfloss = self.lstm_crf.loss(h, attention_mask,labels)

          return lstmcrfloss

        else:

          path_score, best_path = self.lstm_crf.crf(h, attention_mask.bool())
          lstmcrfloss = self.lstm_crf.loss(h, attention_mask, labels)

          return lstmcrfloss,best_path



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


model1 = Decoder()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

 
def train():

    torch.backends.cudnn.enabled=False

    train_dataset = FUNSDDataset(image_file_names=image_files_train, tokenizer=tokenizer, max_length=512, target_size=224)

    train_dataloader = DataLoader(train_dataset, batch_size=4)

    #optimizer = AdamW(model.parameters(), lr=5e-5)
    #lstmcrf的优化器
    optimizer = getattr(optim, "AdamW")
    optimizer = optimizer(model1.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    model1.to(config.device)
    model1.train()


    global_step = 0

    num_train_epochs = 18
    for epoch in range(num_train_epochs):
      print("Epoch:", epoch)

      epoch_loss = 0

      for i,batch in enumerate(train_dataloader):

        input_ids=batch['input_ids'].to(config.device)
        bbox=batch['bbox'].to(config.device)
        attention_mask=batch['attention_mask'].to(config.device)
        token_type_ids=batch['token_type_ids'].to(config.device)
        labels=batch['labels'].to(config.device)
        resized_images = batch['resized_image'].to(config.device) # shape (N, C, H, W), with H = W = 224
        resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(config.device) # single torch tensor that also contains the batch index for every bbox at image size 224

       
        #执行这行代码时，会执行forwrd()函数 
        loss = model1(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
        labels=labels, resized_images=resized_images,
         resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes,

         )

        epoch_loss += loss.item()
        optimizer.zero_grad()
        # backward pass to get the gradients 
        loss.backward()
        # update
        optimizer.step()
        global_step += 1
        #print("第{}个batct的loss为{}".format(i,outputs.loss))
      if(epoch % 2 == 0):
        print("在训练集上第{}个epoch的layoutgat的loss为:{}".format(epoch,epoch_loss))
        test(epoch)

        
  
def test(epoch):

  test_dataset = FUNSDDataset(image_file_names=image_files_test, tokenizer=tokenizer, max_length=512, target_size=224, train=False)
  test_dataloader = DataLoader(test_dataset, batch_size=4,shuffle=True)
  eval_loss = 0.0
  lstmcrf_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None

  #put model in evaluation mode
  model1.to(config.device)
  model1.eval()
  global max_f1score 
  max_f1score = 0

  for batch in tqdm(test_dataloader, desc="Evaluating"):
      with torch.no_grad():
          input_ids=batch['input_ids'].to(config.device)
          bbox=batch['bbox'].to(config.device)
          attention_mask=batch['attention_mask'].to(config.device)
          token_type_ids=batch['token_type_ids'].to(config.device)
          labels=batch['labels'].to(config.device)
          resized_images = batch['resized_image'].to(config.device) 
          resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(config.device) 
        
          # forward pass
          loss,pred = model1(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                          labels=labels, resized_images=resized_images, 
                          resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes,
                          trainer=False)

          # get the loss and logits
          tmp_eval_loss = loss
          eval_loss += tmp_eval_loss.item()
          nb_eval_steps += 1

          # pred.extend([t for t in best_path])
          # true.extend([t for t in tags])

          # compute the predictions
          if preds is None:
              preds = pred.detach().cpu().numpy()
              out_label_ids = labels.detach().cpu().numpy()
          else:
              preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)
              out_label_ids = np.append(
                  out_label_ids, labels.detach().cpu().numpy(), axis=0
              )

  # compute average evaluation loss
  #eval_loss = eval_loss / nb_eval_steps
  
  out_label_list = [[] for _ in range(out_label_ids.shape[0])]
  preds_list = [[] for _ in range(out_label_ids.shape[0])]

  for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
          if out_label_ids[i, j] != 0:
              out_label_list[i].append(idx2label[out_label_ids[i][j]])
              preds_list[i].append(idx2label[preds[i][j]])

  results = {
      "loss": eval_loss,
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list),
      # "每个类别":classification_report(out_label_list,preds_list)
  }
  
  print(results)

  if max_f1score < f1_score(out_label_list, preds_list):
      max_f1score = f1_score(out_label_list, preds_list)

      # if max_f1score > 0.85:

      save_info = {  # 保存的信息
          
            'model': model1.state_dict(),  # 模型的状态字典
          
                }
        # 保存信息
      print("***********{}".format(max_f1score))
      torch.save(save_info, save_model_path+str("%.4f" % max_f1score + '.pth.tar'))

  
  saveLog(epoch=epoch,loss=eval_loss,f1_score="%.4f" % f1_score(out_label_list, preds_list), everyclass = classification_report(out_label_list,preds_list))





def saveLog(epoch,loss,f1_score,everyclass):
  result_dir = "/data/experiments/1data/result/v1modify"
  train_log_filename = "train_log.txt"
  train_log_filepath = os.path.join(result_dir, train_log_filename)
  train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [f1] {f1} [eachclass] {eachclass}\n"

  to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                          epoch = epoch,
                                          loss_str=" ".join(["{}".format(loss)]),
                                          f1 = f1_score, eachclass = everyclass)


  with open(train_log_filepath, "a") as f:
    f.write(to_write)


if __name__ == "__main__":

  #换一下
  setup_seed(2)

  train()