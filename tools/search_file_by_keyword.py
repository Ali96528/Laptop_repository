from ast import keyword
from importlib.metadata import files
import os
from sys import prefix
from tkinter.tix import FileSelectBox


def find_file_by_keywords(path, keyword, postfix):
    """
        搜索带有关键词/后缀的文件
        参数：path-搜索路径目录，关键词，后缀
    """
    files = os.listdir(path)
    for f in files:
        if keyword in f and f.endswith(postfix):   # in 时成员运算符 and 表示并列条件
            print('Found it!'+f)


# 搜索路径
path = 'D:/需要保存的课/黄执中：情绪沟通——改变看法与自我认知'
# 关键词
keyword = '逻辑'
postfix = '.mp4'

find_file_by_keywords(path, keyword, postfix)

"""
    筛选特征:
    1.除了gif以外所有类型
    2.名字中包含关键字'project30'或'commercial'


path = './files'
files = os.listdir(path)
for f in files:
    if (not f.endswith('.gif') and ('project30' in f or 'commercial' in f):
        print(f)
"""
