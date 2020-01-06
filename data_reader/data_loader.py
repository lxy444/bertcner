# -*- coding: utf-8 -*-
from collections import OrderedDict

def load_terminology_dict(filename_dic):
    ''' Load the terminology dicts.
    
    Args:
        filename_dic: such as {'medicine':file1, 'surgery':file2}
    Returns:
        dict. eg. {'medicine':{'扑尔敏'},   'surgery':{'胃癌根治术','直肠癌根治术'}  }
    '''
    out_dic = dict()

    for dic_name_class, filename in filename_dic.items():
        out_dic[dic_name_class] = set()
        f = open(filename, encoding='utf-8')
        for line in f:
            w = line.split()[0]
            out_dic[dic_name_class].add(w)
    return out_dic


def load_radical_dict(filename):
    ''' Load the radical dict.
    
    Args:
        filename: the name of file which stores character and its radical. Each line is 'character, radical'
    Returns:
        dict eg.{char:radical}, radical_lst(no duplication)
    '''    
    out_dict = OrderedDict()
    radical_lst = []  # to ensure the radicals are in the same order
    radical_set = set()
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            try:
                char, radical =  line.strip().split(',')
                out_dict[char] = radical  
                if radical not in radical_set:
                    radical_lst.append(radical)
                    radical_set.add(radical)      
            except:
                print(f'Radical loading skip: {line}')
    return out_dict, radical_lst


def load_label_list(filename):
    ''' Load the labels
    
    Args:
        filename: the file name which contains a label in each line
    Returns:
        label_list
    '''
    out_lst = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            out_lst.append(line.strip())
    return out_lst
