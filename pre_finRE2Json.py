import json
import os
import pickle
import numpy as np

str2 = "深圳光电	京东方	分析	近日,在“<N>深圳光电显示周暨深圳彩电节”期间,京东方正式发布AM-OLED发展规划,积极布局新型显示技术。"

arrs = str2.split()
print(arrs)
print(type(arrs))
subj = arrs[0]
obj = arrs[1]
relation = arrs[2]
sentence = arrs[3]
subj_start = sentence.index(subj)
subj_end = subj_start+len(subj)
obj_start = sentence.index(obj)
obj_end = obj_start+len(obj)
print(subj)
print(obj)
print(relation)
print(sentence)
word_list = str(list(sentence))
print(word_list)
word_list = word_list.replace("'",'"')
print(word_list)
id = 888
# str4 = "{ "id": "1123", "relation": "Use", "token": [ "阿", "纤", "在", "蒲", "老", "前", "辈", "的", "笔", "下", "很", "是", "可", "爱" ], "stanford_pos": null, "stanford_head": null, "stanford_deprel": null, "subj_start": 3, "subj_end": 6, "obj_start": 8, "obj_end": 8, "subj_type": "", "obj_type": "" }"
# str4 = '{"accessToken": "'+str(id)+'", "User-Agent": "Apache-HttpClient/4.5.2 (Java/1.8.0_131)"}'
str4 = '{ "id": "'+str(id)+'", "relation": "'+relation+'", "token": '+word_list+', "stanford_pos": null, "stanford_head": null, "stanford_deprel": null, "subj_start": '+str(subj_start)+', "subj_end": '+str(subj_end)+', "obj_start": '+str(obj_start)+', "obj_end": '+str(obj_end)+', "subj_type": "", "obj_type": "" }'
print(str4)
print(json.loads(str4))

# 输出
# {'accessToken': '521de21161b23988173e6f7f48f9ee96e28', 'User-Agent': 'Apache-HttpClient/4.5.2 (Java/1.8.0_131)'}
# <class 'dict'>

# str4 = "{'id': '1123', 'relation': 'Use', 'token': ['阿', '纤', '在', '蒲', '老', '前', '辈', '的', '笔', '下', '很', '是', '可', '爱'], 'stanford_pos': None, 'stanford_head': None, 'stanford_deprel': None, 'subj_start': 3, 'subj_end': 6, 'obj_start': 8, 'obj_end': 8, 'subj_type': '', 'obj_type': ''}"
# arrs = str4.split(',')
# print(arrs)
# str5 = ''
# for i in arrs:
#     str5+=i
#     str5+='\n'
#
# print(str5)

def read_text(fname):
    text = ''
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    id = -1
    text_json = '['
    print("len = ",len(lines))
    for i in range(0, len(lines),2):
        print(lines[i].strip())
        line = lines[i].strip()

        arrs = line.split('\t')
        subj = arrs[0]
        obj = arrs[1]
        relation = arrs[2]
        sentence = arrs[3]
        print(sentence)
        subj_start = sentence.index(subj)
        subj_end = subj_start + len(subj)
        obj_start = sentence.index(obj)
        obj_end = obj_start + len(obj)
        word_list = str(list(sentence))
        word_list = word_list.replace("'", '"')
        id+=1
        str4 = '{ "id": "' + str(
            id) + '", "relation": "' + relation + '", "token": ' + word_list + ', "stanford_pos": null, "stanford_head": null, "stanford_deprel": null, "subj_start": ' + str(
            subj_start) + ', "subj_end": ' + str(subj_end) + ', "obj_start": ' + str(obj_start) + ', "obj_end": ' + str(
            obj_end) + ', "subj_type": "", "obj_type": "" }'
        print(str4)
        # print(json.loads(str4))
        if id < len(lines)/2-1:
            str4 += ','
        text_json+=str4
        text_json+='\n'
        # print(json.loads(str4))
    text_json = text_json.strip(',')
    text_json+=']'
    print(text_json)
    print(json.loads(text_json))

    fname = fname.replace('txt','json')
    with open(fname, 'a', encoding='utf-8', newline='\n') as f:
        f.write(text_json)
        # f.write(aspect + '\n')
        # f.write(polarity + '\n')

if __name__ == '__main__':
    fname =  {
    'train': './dataset/FinRE/train.txt',
    'test': './dataset/FinRE/test.txt',
    'dev': './dataset/FinRE/dev.txt'
    }
    # print(fname['dev'])
    read_text(fname['train'])
