import os
import json
from PIL import Image, ImageDraw, ImageFont
# image_files_path = "/home/litang/LtProject/gongsi/training_data/images"
image_files_path = "/home/litang/LtProject/sample/images"
image_files = os.listdir(image_files_path)
image_files = sorted(image_files)
# jsonPath = '/home/litang/LtProject/gongsi/training_data/annotations'
jsonPath = '/home/litang/LtProject/sample/annotations'
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
save_image_path= "/home/litang/LtProject/sample/truima"

def saveImage(image,image_name):
  image.save(os.path.join(save_image_path,image_name))

fontpath = "/home/litang/LtProject/layout/font/MSYHL.TTC"
textsize=15
font = ImageFont.truetype(fontpath,textsize)

for i in image_files:
    image = Image.open(os.path.join(image_files_path,i))
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    with open(os.path.join(jsonPath,i.split('.')[0]+'.json')) as f:
        data = json.load(f)
    count=0
    for annotation in data:
        label = annotation['label']
        general_box = annotation['position']
        if label in list(label2color.keys()):
            draw.rectangle(general_box, outline=label2color[label], width=2)
            draw.text((general_box[0] + 10, general_box[1] - 10), label + '第'+str(count), fill=label2color[label], font=font)
        else:
            continue
        #   words = annotation['']
        count+=1
    saveImage(image,i)

#   for word in words:
#     box = word['box']
#     draw.rectangle(box, outline=label2color[label], width=1)
print("a")