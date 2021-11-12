import os
import json
import random
import xml.etree.ElementTree as ET

def search(pattern, sequence):
    """从sequence中寻找子串pattern
       如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding='utf-8') as outfile:
        [outfile.write(d+'\n') for d in data]

def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

def get_file_list(path):
    # 根目录路径
    # root_path = r""
    root_path = path
    # 用来存放所有的文件路径
    file_list = []
    # 用来存放所有的目录路径
    dir_list = []
    get_file_path(root_path, file_list, dir_list)
    return file_list

file_list = get_file_list('./COERKB/')

count = 0
output = []
tempDict = {}
for file_path in file_list:
    # print(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()
    # 标签名
    # print('root_tag:',root.tag)
    for sro in root:
        # 标签中内容
        text = sro[0].text
        s = sro[1][0].text
        r = sro[2].attrib["phrase_text"]
        o = sro[1][1].text
        tempList = []
        if text not in tempDict:
            tempDict[text] = []
        tempDict[text].append((s,r,o))
for k, v in tempDict.items():
    tempDict1 = {}
    tempList1 = []
    tempList2 = []
    for s, r, o in v:
        tempList2.append((s, r, o))
    for sro in set(tempList2):
        tempDict2 = {}
        subject_id = search(sro[0], k)
        relation_id = search(sro[1], k)
        object_id = search(sro[2], k)
        if subject_id != -1 and relation_id != -1 and object_id != -1 and object_id>relation_id and relation_id>subject_id:
            tempDict2['subject'] = sro[0]
            tempDict2['relation'] = sro[1]
            tempDict2['object'] = sro[2]
            tempList1.append(tempDict2)
    if len(tempList1) > 1 and len(tempList1) <= 4:
        count += 1
        tempDict1['text'] = k.replace(' ', '')
        tempDict1['sro_list'] = tempList1
        l1 = json.dumps(tempDict1, ensure_ascii=False)
        output.append(l1)

print(count)
random.seed(2021)# 设置随机种子
random.shuffle(output)  # 随机一下
# 按照 8 / 2 分
train_data_len = int(len(output) * 0.8)


train_data = output[:train_data_len]
valid_data = output[train_data_len:]

save_dir = './'
write_by_lines(u"{}/train.json".format(save_dir), train_data)
write_by_lines(u"{}/dev.json".format(save_dir), valid_data)