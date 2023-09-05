import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '..')))

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import random
import numpy as np
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertTokenizer
from transformers import LayoutLMv2ForTokenClassification, AdamW, LayoutLMv2Processor
from os import listdir
# from transformers import AdamW
from tqdm.notebook import tqdm
# from transformers import LayoutLMv2Processor
import time
import os
from config import Config
config = Config()
if config.use_en:
  tokenizer = BertTokenizer.from_pretrained('/home/litang/LtProject/FAEA-FSRC-main/pretrain/bert-base-uncased/bert-base-uncased-vocab.txt')
else:
  tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
processor = LayoutLMv2Processor.from_pretrained("/home/litang/LtProject/FAEA-FSRC-main/pretrain/layoutlmv2-base-uncased", revision="no_ocr")
# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

# from layoutgcn.test_01 import EPOCHS
save_model_path = "/data/experiments/1data/savedmodel/Augmentv2/"
# image_files_train = [f for f in listdir('/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/dataset_normal/training_data/images')]
image_files_train = [f for f in listdir('/home/litang/LtProject/gongsi/training_data/images')]

# list all test image file names
# image_files_test = [f for f in listdir('/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/dataset_normal/testing_data/images')]
image_files_test = [f for f in listdir('/home/litang/LtProject/gongsi/testing_data/images')]

# labels = ['B-answer', 'I-answer','B-header','I-header', 'B-question','I-question', "B-other","I-other"]
# labels
# idx2label = {v: k for v, k in enumerate(labels)}
# label2idx = {k: v for v, k in enumerate(labels)}
# label2idx
from custon_datasetwithoutpad import idx2label,label2idx, get_needed_key
value_list = [value + "-value" for value in list(get_needed_key().keys())]
allclass = list(get_needed_key().keys()) +['0'] + value_list

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

    return [x - 0.5, y - 0.5, xmax + 0.5, ymax + 0.5]


class FUNSDDataset(Dataset):
    """LayoutLM dataset with visual features."""

    def __init__(self, image_file_names, max_length,target_size, train=True, predict=False):
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
            base_path = "/home/litang/LtProject/gongsi/training_data"
        elif self.predict:
            base_path = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/layoutlm_data"
        else:
            base_path = "/home/litang/LtProject/gongsi/testing_data"
            # base_path = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/layoutlm_data"
        original_image = Image.open(base_path + "/images/" + item).convert("RGB")
        # resize to target size (to be provided to the pre-trained backbone)
        resized_image = original_image.resize((self.target_size, self.target_size))

        # first, read in annotations at word-level (words, bounding boxes, labels)
        with open(base_path + '/annotations/' + item[:-4] + '.json') as f:
            data = json.load(f)
        words = []
        unnormalized_word_boxes = []
        word_labels = []
        # for annotation in data['form']:
        #     # get label
        #     label = annotation['label']
        #     # get words,annotation['words']是一个列表，里面可能有多个字典，每个字典中有'text'项和'box项'

        #     if annotation['text'] == '':
        #         continue
        #     words.append(annotation['text'])
        #     unnormalized_word_boxes.append(annotation['box'])
        #     word_labels.append(label)
        for annotation in data:
          if annotation['label'] in allclass:
            label = annotation['label']
            if annotation['label'] in ['交货地-value', '卸货港-value', '收货地-value', '目的港-value','起运港-value']:
                annotation['text'] = random.choice(list_address)                
          # get label
          else:
            label = '0'
          # get words,annotation['words']是一个列表，里面可能有多个字典，每个字典中有'text'项和'box项'
          
          if annotation['text'] == '':
            continue

          words.append(annotation['text'])
          unnormalized_word_boxes.append(annotation['position'])
          word_labels.append(label)

        width, height = original_image.size
        # 经过normalize_box后四个坐标中第2个和第4个没变，是因为图像的height有的为1000，这样分子分母都是1000，就抵消了
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        
        for box in normalized_word_boxes:

          for i in range(len(box)):
            if box[i] > 1000:
              box[i] = 1000
            if box[i] < 0:
              box[i] = 0

        assert len(words) == len(normalized_word_boxes)

        # next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
        # token_boxes中存放的是 经过 normalize_box()函数处理过的坐标
        token_boxes = []

        # unnormalized_token_boxes存放的是原始的未经处理的坐标，后面的 esize_and_align_bounding_box()函数会用
        unnormalized_token_boxes = []
        token_labels = []

        for word, unnormalized_box, box, label in zip(words, unnormalized_word_boxes, normalized_word_boxes,
                                                      word_labels):

            # 对于ocr得到的结果的json文件中, 除了Date:这种，其他的一般都已经是单个单词了，所以这里经过self.tokenizer.tokenize得到的结果大多数 都还是原来的token，
            # 只有Date:这种 被分为 ["Date",":"]2个token
            word_tokens = self.tokenizer.tokenize(word)
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
            # label first token as B-label (beginning), label all remaining tokens as I-label (inside)
            
            for i in range(len(word_tokens)):
                  if i == 0:
                    token_labels.extend(['B-' + label])
                  else:
                    token_labels.extend(['I-' + label])
            

        # Truncation of token_boxes + token_labels
        special_tokens_count = 2
        if len(token_boxes) > self.max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
            token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]

        # add bounding boxes and labels of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
        token_labels = [-100] + token_labels + [-100]

        # 新添加的语句，可能是因为transformers的版本问题。因为设置了paddig="max_length"，所以input_ids,attention_mask,token_type_ids的长度都是512
        encoding = self.tokenizer(text=' '.join(words),
                                  padding="max_length",
                                  max_length=512, 
                                  truncation=True)

        # encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        # 注释掉原有的语句
        # Padding of token_boxes up the bounding boxes to the sequence length.

        # 获取实际长度，为了下面添加token_boxes和token_labels属性，并使得其长度也为512
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = self.max_seq_length - len(input_ids)
        token_boxes += [self.pad_token_box] * padding_length
        unnormalized_token_boxes += [self.pad_token_box] * padding_length
        token_labels += [-100] * padding_length

        # 添加token_boxes和token_labels属性
        encoding['bbox'] = token_boxes
        encoding['labels'] = token_labels
        encoding['unnormalized_token_boxes'] = unnormalized_token_boxes
        assert len(encoding['input_ids']) == self.max_seq_length  # [512]

        assert len(encoding['attention_mask']) == self.max_seq_length  # []
        assert len(encoding['token_type_ids']) == self.max_seq_length
        assert len(encoding['bbox']) == self.max_seq_length  # [512,4]
        assert len(encoding['labels']) == self.max_seq_length  # [512]

        encoding['image'] = ToTensor()(resized_image)
        # rescale and align the bounding boxes to match the resized image size (typically 224x224)
        # 添加resized_and_aligned_bounding_boxes属性,长度也为512，因为unnormalized_token_boxes已经经过了padding
        # finally, convert everything to PyTorch tensors
        for k, v in encoding.items():
            # 把label字符串转化为labletoindex中对应的索引
            if k == 'labels':
                label_indices = []
                # convert labels from string to indices
                for label in encoding[k]:
                    if label != -100:
                        label_indices.append(label2idx[label])
                    else:
                        label_indices.append(label)
                encoding[k] = label_indices
            # 把encoding中的其他属性也 由list转化为 tensor
            encoding[k] = torch.as_tensor(encoding[k])

        return encoding



# 定义模型
# So if we want to define a model that includes LayoutLM + the visual embeddings, it would look like this:


class LayoutLMForTokenClassification(nn.Module):
    def __init__(self, output_size=(3, 3),
                 spatial_scale=14 / 224,
                 sampling_ratio=2
                 ):
        super().__init__()

        # LayoutLM base model + token classifier
        self.num_labels = len(label2idx)  # 8
        self.layoutlm = LayoutLMv2ForTokenClassification.from_pretrained('/home/litang/LtProject/FAEA-FSRC-main/pretrain/layoutlmv2-base-uncased',
                                                                       num_labels=self.num_labels)
        # self.layoutlm = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
        #                                                                num_labels=self.num_labels)
    def forward(
            self,
            input_ids,  # [batchsize,max_length]
            attention_mask,
            token_type_ids,
            bbox,
            images,
            labels      
        
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        """        
        # first, forward pass on LayoutLM
        outputs = self.layoutlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox,
            image=images,
            labels=labels
        )  # odict_keys(['last_hidden_state', 'pooler_output'])

        return outputs

model = LayoutLMForTokenClassification()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train():
    train_dataset = FUNSDDataset(image_file_names=image_files_train, max_length=512,
                                 target_size=224)

    train_dataloader = DataLoader(train_dataset, batch_size=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    torch.backends.cudnn.enabled = False

    model.to(device)
    model.train()

    global_step = 0

    num_train_epochs = 20
    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            images = batch['image'].to(device)  # shape (N, C, H, W), with H = W = 224
            # single torch tensor that also contains the batch index for every bbox at image size 224

            # 执行这行代码时，会执行forwrd()函数
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            bbox=bbox,
                            images=images,
                            labels=labels
                           )
            # outputs = outputs[0]
            loss = outputs.loss
            # loss = outputs[0].loss
            epoch_loss += loss

            optimizer.zero_grad()
            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()

            global_step += 1

        if (epoch % 2 == 0):
            print("在训练集上第{}个epoch的loss为:{}".format(epoch, epoch_loss))

            test(epoch)
            # print("第{}个batct的loss为{}".format(i,outputs.loss))


import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def test(epoch):
    test_dataset = FUNSDDataset(image_file_names=image_files_test, max_length=512, target_size=224,
                                train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    # put model in evaluation mode
    model.to(device)
    model.eval()
    torch.backends.cudnn.enabled = False
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            images = batch['image'].to(device)
            # forward pass
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            bbox=bbox,
                            images=images,
                            labels=labels
                            )

            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    #eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(idx2label[out_label_ids[i][j]])
                preds_list[i].append(idx2label[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        # "每个类别":classification_report(out_label_list,preds_list)
    }

    if f1_score(out_label_list, preds_list) > 0.86:
        save_info = {  # 保存的信息
        
        'model': model.state_dict(),  # 模型的状态字典
        
            }
    # 保存信息
        torch.save(save_info, save_model_path+str("%.4f" % f1_score(out_label_list, preds_list)) + '.pth.tar')
    print(results)
    # print(classification_report(out_label_list,preds_list))

    
    
    saveLog(epoch=epoch,loss=eval_loss,f1_score="%.4f" % f1_score(out_label_list, preds_list), everyclass = classification_report(out_label_list,preds_list))



def saveLog(epoch,loss,f1_score,everyclass):
  result_dir = "/data/experiments/1data/result/Augmentv2"
  train_log_filename = "layoutLMV2_aug_train_log.txt"
  train_log_filepath = os.path.join(result_dir, train_log_filename)
  train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [f1] {f1} [eachclass] {eachclass}\n"

  to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                          epoch = epoch,
                                          loss_str=" ".join(["{}".format(loss)]),
                                          f1 = f1_score, eachclass = everyclass)


  with open(train_log_filepath, "a") as f:
    f.write(to_write)

if __name__ == "__main__":
    train()
    print("aa")



#save_info = { 'model': model.state_dict(),}