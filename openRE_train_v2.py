#! -*- coding:utf-8 -*-
# 开放领域关系抽取任务，基于GPLinker

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json
import numpy as np
from itertools import groupby
from bert4keras.backend import keras, K
from bert4keras.backend import sparse_multilabel_categorical_crossentropy
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from tqdm import tqdm

maxlen = 128
batch_size = 16
epochs = 10
config_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

# 读取schema
label = ["subject", "relation", "object"]
labels = []
for i in range(1, 5):
    labels.append(("{}".format(i), "subject"))
    labels.append(("{}".format(i), "relation"))
    labels.append(("{}".format(i), "object"))
print(labels)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
       如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def load_data(filename):
    """加载数据
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = {'text': l['text'], 'sro_list': []}
            for i, sro in enumerate(l['sro_list']):
                tempList = []
                index = i + 1
                idx_subject = search(sro["subject"], l["text"])
                idx_relation = search(sro["relation"], l["text"])
                idx_object = search(sro["object"], l["text"])
                tempList.append((str(index), "subject", sro["subject"], idx_subject))
                tempList.append((str(index), "relation", sro["relation"], idx_relation))
                tempList.append((str(index), "object", sro["object"], idx_object))
                d["sro_list"].append(tempList)
            D.append(d)
    return D


# 加载数据集
train_data = load_data('./data/CORE/train.json')
valid_data = load_data('./data/CORE/dev.json')
print("train_data:", len(train_data))
print("valid_data:", len(valid_data))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d['text'], maxlen=maxlen)
            mapping = tokenizer.rematch(d['text'], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            # 整理事件
            events = []
            for e in d['sro_list']:
                events.append([])
                for t, r, a, i in e:
                    label = labels.index((t, r))
                    start, end = i, i + len(a) - 1
                    if start in start_mapping and end in end_mapping:
                        start, end = start_mapping[start], end_mapping[end]
                        events[-1].append((label, start, end))
            # 构建标签
            argu_labels = [set() for _ in range(len(labels))]
            head_labels, tail_labels = set(), set()
            for e in events:
                for l, h, t in e:
                    argu_labels[l].add((h, t))
                for i1, (_, h1, t1) in enumerate(e):
                    for i2, (_, h2, t2) in enumerate(e):
                        if i2 > i1:
                            head_labels.add((min(h1, h2), max(h1, h2)))
                            tail_labels.add((min(t1, t2), max(t1, t2)))
            for label in argu_labels + [head_labels, tail_labels]:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            argu_labels = sequence_padding([list(l) for l in argu_labels])
            head_labels = sequence_padding([list(head_labels)])
            tail_labels = sequence_padding([list(tail_labels)])
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_argu_labels.append(argu_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_argu_labels = sequence_padding(batch_argu_labels, seq_dims=2)
                batch_head_labels = sequence_padding(batch_head_labels, seq_dims=2)
                batch_tail_labels = sequence_padding(batch_tail_labels, seq_dims=2)
                yield [batch_token_ids, batch_segment_ids], [
                    batch_argu_labels, batch_head_labels, batch_tail_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis=1))


# 加载预训练模型
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False
)
output = base.model.output

# 预测结果
argu_output = GlobalPointer(heads=len(labels), head_size=64)(output)
head_output = GlobalPointer(heads=1, head_size=64, RoPE=False)(output)
tail_output = GlobalPointer(heads=1, head_size=64, RoPE=False)(output)
outputs = [argu_output, head_output, tail_output]

# 构建模型
model = keras.models.Model(base.model.inputs, outputs)
model.compile(loss=globalpointer_crossentropy, optimizer=Adam(2e-5))
# model.summary()


class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）
    """
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def extract_events(text, threshold=0):
    """抽取输入text所包含的所有事件
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取论元
    argus = set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        argus.add(labels[l] + (h, t))
    # print("argus:", argus)
    # 构建链接
    links = set()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if outputs[1][0, min(h1, h2), max(h1, h2)] > threshold:
                    if outputs[2][0, min(t1, t2), max(t1, t2)] > threshold:
                        links.add((h1, t1, h2, t2))
                        links.add((h2, t2, h1, t1))
    # print("links:", links)
    # 析出事件
    events = []
    for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
        for event in clique_search(list(sub_argus), links):
            events.append([])
            for argu in event:
                start, end = mapping[argu[2]][0], mapping[argu[3]][-1] + 1
                events[-1].append(argu[:2] + (text[start:end], start))
    return events


def evaluate(data, threshold=0):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=0):
        sro_pred = []
        sro_y = []
        pred_sros = extract_events(d['text'], threshold)
        for sros in pred_sros:
            tempDict = {}
            for sro in sros:
                if sro[1] == "subject":
                    tempDict["subject"] = sro[2]
                if sro[1] == "relation":
                    tempDict["relation"] = sro[2]
                if sro[1] == "object":
                    tempDict["object"] = sro[2]
            if "subject" in tempDict and "relation" in tempDict and "object" in tempDict:
                sro_pred.append((tempDict["subject"], tempDict["relation"], tempDict["object"]))
        sro_pred = list(set(sro_pred))#去重

        for sros in d['sro_list']:
            tempDict = {}
            for sro in sros:
                if sro[1] == "subject":
                    tempDict["subject"] = sro[2]
                if sro[1] == "relation":
                    tempDict["relation"] = sro[2]
                if sro[1] == "object":
                    tempDict["object"] = sro[2]
            if "subject" in tempDict and "relation" in tempDict and "object" in tempDict:
                sro_y.append((tempDict["subject"], tempDict["relation"], tempDict["object"]))

        R = set([SPO(spo) for spo in sro_pred])
        T = set([SPO(spo) for spo in sro_y])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z


    return f1, precision, recall


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        # print('spo: ' + str(spo))
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model_open_sro.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f'
            % (f1, precision, recall, self.best_val_f1)
        )

if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
