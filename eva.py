from os import listdir
import json
json_list = listdir("/home/litang/LtProject/haiyundan/v2/savejson")
ground_truth = []
predict = []
for i in json_list:
    contrast_file_path = '/home/litang/LtProject/haiyundan/model_mark_json/' + i.split(".")[0] +".json"
    predict_file_path = '/home/litang/LtProject/haiyundan/v2/savejson/' + i.split(".")[0] +".json"
    with open(contrast_file_path) as f:
        datag = json.load(f)
    # ground_truth = []
    for annotation in datag:
        ground_truth.append(annotation['label'])
    with open(predict_file_path) as f:
        datap = json.load(f)
    # ground_truth = []
    for annotation in datap:
        predict.append(annotation['label'][0])
from custon_datasetwithoutpad import idx2label,label2idx, get_needed_key
value_list = [value + "-value" for value in list(get_needed_key().keys())]
allclass = list(get_needed_key().keys()) +['0'] + value_list
need_remove_idx = []
for i, value in enumerate(ground_truth):
    if value not in allclass:
        need_remove_idx.append(i)
for i in range(len(need_remove_idx)):
    del ground_truth[need_remove_idx[len(need_remove_idx)-i-1]]
    del predict[need_remove_idx[len(need_remove_idx)-i-1]]
predict= [predict]
ground_truth=[ground_truth]
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
results = {
        "precision": precision_score(predict, ground_truth),
        "recall": recall_score(predict, ground_truth),
        "f1": f1_score(predict, ground_truth),
        # "每个类别":classification_report(predict, ground_truth)
    }
print(results)
print(classification_report(predict, ground_truth))
print("a")