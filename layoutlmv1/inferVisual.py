#-----inferrenceAndVisualization----
import sys
from os.path import normpath,join,dirname
from matplotlib import collections
sys.path.append(normpath(join(dirname(__file__), '..')))
from PIL import Image, ImageDraw, ImageFont
import torch
from collections import Counter
import os
from config import Config
config = Config()
from layout_v1 import FUNSDDataset,idx2label,model
from transformers import BertTokenizer

if config.use_en:
  tokenizer = BertTokenizer.from_pretrained('/home/litang/LtProject/FAEA-FSRC-main/pretrain/bert-base-uncased/bert-base-uncased-vocab.txt')
else:
  tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model_path = "/data/experiments/1data/savedmodel/layoutlmv1/0.8504.pth.tar"
#测试的图片路径
image_files_path = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/layoutlm_data/images"

#保存图片的路径
save_image_path= "/home/litang/LtProject/haiyundan/v1/saveimage"

#测试的json文件路径
jsonPath = "/home/zhangjunwei/Zjwproject/layoutgcn1/haiyundan/layoutlm_data/annotations"
#保存的json文件路径
savePath = "/home/litang/LtProject/haiyundan/v1/savejson"

image_files = os.listdir(image_files_path)
# image_files = ["15705.png","15706.png",'15710.png','15714.png','15715.png','15788.png','15790.png']


#加载模型参数
#加载模型参数

state_dict = torch.load(model_path)
model.load_state_dict(state_dict['model'])
model.to(config.device)
#标签
label2color = {'发货人': 'blue', '收货人': 'green', '通知人': 'orange', '航次': 'red', 
'船名': 'blue', '收货地': 'green', '起运港': 'orange', '卸货港': 'red', '目的港': 'blue', 
'运输条款': 'green', '运费条款': 'orange', '船公司': 'red', '约号': 'blue', '预配船期': 'green',
 '箱型箱量': 'orange', '货品种类': 'red', 'HSCODE': 'blue', '委托公司': 'green', '备注': 'orange',
  '客户委托编码': 'red', '交货地': 'blue', '货好时间': 'green', '贸易条款': 'orange', '出单方式': 'red',
   '唛头': 'blue', '件数': 'green', '毛重': 'orange', '体积': 'red', '品名': 'blue',
    '电话': 'green', '发货人-value': 'orange', '收货人-value': 'red', '通知人-value': 'blue',
     '航次-value': 'green', '船名-value': 'orange', '收货地-value': 'red', '起运港-value': 'blue', 
     '卸货港-value': 'green', '目的港-value': 'orange', '运输条款-value': 'red', '运费条款-value': 'blue',
      '船公司-value': 'green', '约号-value': 'orange', '预配船期-value': 'red', '箱型箱量-value': 'blue', 
      '货品种类-value': 'green', 'HSCODE-value': 'orange', '委托公司-value': 'red', '备注-value': 'blue', 
      '客户委托编码-value': 'green', '交货地-value': 'orange', '货好时间-value': 'red', '贸易条款-value': 'blue', 
      '出单方式-value': 'green', '唛头-value': 'orange', '件数-value': 'red', '毛重-value': 'blue', 
'体积-value': 'green', '品名-value': 'orange', '电话-value': 'red','0':"blue"}


def dealAllImage():

  
  for i in range(len(image_files)):

    image_file_name = []
    image_file_name.append(image_files[i])
    inference_dataset = FUNSDDataset(image_file_names=image_file_name, tokenizer=tokenizer, max_length=512, target_size=224, train=False, predict = True)
    test_encoding = inference_dataset[0]
    test_encoding.keys()
    dealOneImage(test_encoding,image_files[i])
  
def iob_to_label(label):
    return label[2:]


def saveImage(image,image_name):
  image.save(os.path.join(save_image_path,image_name))


import json
def dealOneImage(test_encoding,image_name):

  with open(os.path.join(jsonPath,image_name.split('.')[0]+'.json')) as f:
    data = json.load(f)

  for item in data:
    item["label"] = []

  image = Image.open(os.path.join(image_files_path,image_name))
  image = image.convert("RGB")

  for k,v in test_encoding.items():
    test_encoding[k] = test_encoding[k].unsqueeze(0).to(config.device)

  input_ids=test_encoding['input_ids']
  bbox=test_encoding['bbox']
  attention_mask=test_encoding['attention_mask']
  token_type_ids=test_encoding['token_type_ids']
  labels=test_encoding['labels']
  resized_images = test_encoding['resized_image']
  resized_and_aligned_bounding_boxes = test_encoding['resized_and_aligned_bounding_boxes']

  # forward pass to get logits
  outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                labels=labels, resized_images=resized_images,
                 resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes,
                )
  token_predictions = outputs.logits.argmax(-1).squeeze().tolist() # the predictions are at the token level

  token_actual_boxes = test_encoding['unnormalized_token_boxes'].squeeze().tolist()

  word_level_predictions = [] # let's turn them into word level predictions
  final_boxes = []
  for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
    if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, 
                                                            tokenizer.sep_token_id, 
                                                           tokenizer.pad_token_id]):
      # skip prediction + bounding box
      continue
    else:
      word_level_predictions.append(token_pred)
      final_boxes.append(box)


  draw = ImageDraw.Draw(image)

  fontpath = "/home/litang/LtProject/layout/font/MSYHL.TTC"
  textsize=15
  font = ImageFont.truetype(fontpath,textsize)

  for prediction, box in zip(word_level_predictions, final_boxes):

    predicted_label = iob_to_label(idx2label[prediction])
    for item in data:
      if([int(item) for item in item["position"]] == [int(item )for item in box]):
        item["label"].append(predicted_label)
        break
  #   draw.rectangle(box, outline=label2color[predicted_label])
    
  #   draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

  # # image
  # saveImage(image,image_name)

  for item in data:
    # item["label"] = list(set(item["label"]))
    if len(item["label"]) != 0:
      collection_labels = Counter(item["label"])
      item["label"] = []
      item["label"].append(collection_labels.most_common(1)[0][0])
      draw.rectangle(item["position"], outline=label2color[item["label"][0]])
      draw.text((item["position"][0] + 10, item["position"][1] - 10), text=item["label"][0], fill=label2color[item["label"][0]], font=font)
    else:
      item["label"] = ['0']
      draw.rectangle(item["position"], outline=label2color[item["label"][0]])
      draw.text((item["position"][0] + 10, item["position"][1] - 10), text=item["label"][0], fill=label2color[item["label"][0]], font=font)
  saveImage(image,image_name)
  with open(os.path.join(savePath,image_name.split('.')[0]+'.json'),'w',encoding="utf8") as f:

    json.dump(data,f,ensure_ascii=False,indent=2)


dealAllImage()

print("aa")
#-----------------------------------------v3(end)--------------------------------


