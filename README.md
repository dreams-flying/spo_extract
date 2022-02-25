# spo_extract
本项目是利用深度学习技术进行开放领域的关系抽取，算法模型可以处理多个三元组。</br></br>
开放领域关系抽取是从文本中抽取三元组，不同于限定域的关系抽取，开放关系抽取并不限制关系的种类和数量，即所识别的关系(relation)在文本中体现。于是将("n", 标签)组合成一个大类，然后可以将开放关系抽取转化为一个NER问题，其中n对应的是文本中三元组的数量。对于NER任务，我们采用[GPLinker模型](https://kexue.fm/archives/8926)，相应代码在openRE_train_v2中。
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.9</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── data    存放数据</br>
├── pretrained_model    存放预训练模型</br>
├── extract_chinese_and_punct.py</br>
├── openRE_train.py    训练代码</br>
├── openRE_train_v2.py    训练代码v2</br>
├── openRE_predict.py    评估和测试代码</br>
# 数据集
采用[COER语料库](https://github.com/TJUNLP/COER)，对原始数据进行了筛选，处理好的数据存放在data/CORE文件夹下。</br>
```
"text": "巴基斯坦国家灾害管理局局长法鲁克、巴内阁事务部长穆罕默德", 
"sro_list": [{"subject": "巴", "relation": "内阁事务部长", "object": "穆罕默德"}, {"subject": "巴基斯坦国家灾害管理局", "relation": "局长", "object": "法鲁克"}]
```
三元组中的实体和关系均出现在文本中。
训练集和验证集中的每个句子都有n个三元组(2=<n<=4)，数据统计情况：
| 数据集 | 2个 | 3个 | 4个 |
| :------:| :------: | :------: | :------: |
| train | 8858 | 767 | 264 |
| dev | 2238 | 177 | 58 |
# 使用说明
1.[下载预训练语言模型](https://github.com/google-research/bert#pre-trained-models)</br>
  可采用BERT-Base, Chinese等模型</br>
  更多的预训练语言模型可参见[bert4keras](https://github.com/bojone/bert4keras)给出的权重。</br>
2.构建数据集(数据集已处理好)</br>
  train.json和dev.json</br>
3.训练模型
```
python openRE_train.py
```
4.评估和测试
```
python openRE_predict.py
```
# 结果
| 数据集 | f1 | precision | recall |
| :------:| :------: | :------: | :------: |
| train | 0.92781 | 0.92947 | 0.92616 |
| dev | 0.62125 | 0.61854 | 0.62397 |
| dev_v2 | 0.76922 | 0.80624 | 0.73545 |
