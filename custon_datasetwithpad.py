def get_needed_key():
    """
       配置需要匹配的字段
    """
    return {
        "发货人": "",
        "收货人": "",
        "通知人": "",
        "航次": "",
        "船名": "",
        "收货地": "",
        "起运港": "",
        "卸货港": "",
        "目的港": "",
        "运输条款": "",
        "运费条款": "",
        "船公司": "",
        "约号": "",
        "预配船期": "",
        "箱型箱量": "",
        "货品种类": "",
        "HSCODE": "",
        "委托公司": "",
        "备注": "",
        "客户委托编码": "",
        "交货地": "",
        "货好时间": "",
        "贸易条款": "",
        "出单方式": "",
        "唛头": "",
        "件数": "",
        "毛重": "",
        "体积": "",
        "品名": "",
        "电话": ""
    }

# value_labels = [value + "-value" for value in list(get_needed_key().keys())]
# key_labels = [key for key in list(get_needed_key().keys())]

b_key_labels = ["B-" + key for key in list(get_needed_key().keys())]
i_key_labels = ["I-" + key for key in list(get_needed_key().keys())]
b_value_labels = ["B-" + value + "-value" for value in list(get_needed_key().keys())]
i_value_labels = ["I-" + value + "-value" for value in list(get_needed_key().keys())]

labels = b_key_labels \
         + i_key_labels \
         + b_value_labels \
         + i_value_labels \
         + ["B-0","I-0"]

labels = ['O'] + labels

idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}



