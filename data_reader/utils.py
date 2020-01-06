# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sys
sys.path.append('..')
from data_reader.BiMM import BiMM


def padding(lst, max_seq_len, pad_item):
    '''Padding for lst.
        
    Args:
        lst: a list
        max_seq_len: max sequence_length
        pad_item: the item to pad
    
    Returns:
        a list
    '''
    if len(lst)>=max_seq_len:
        return lst[:max_seq_len]
    else:
        lst += [pad_item] * (max_seq_len-len(lst))
        return lst
    
def one_hot(tensor_data, num_classes):
    ''' Map the tensor data to one-hot encoding
    
    Args:
        tensor_data: shape (batch_size, max_seq_len)
        num_classes: the number of classes for one-hot encoding
    '''
    # the function code ref:https://blog.csdn.net/qq_34914551/article/details/88700334

    ## edit: 
    if torch.cuda.is_available():
        tensor_data = tensor_data.cpu()

    size = list(tensor_data.size())
    tensor_data = tensor_data.view(-1)   
    ones = torch.eye(num_classes)
    ones = ones.index_select(0, tensor_data)   
    size.append(num_classes) 
    return ones.view(*size)

   
def find_the_label_never_go(label,tag2idx):
    """ For a given label, find the labels it can never go to.
    
    Args:
        label: label
        tag2idx: {tag: id}
    Returns:
        the list of labels
    """
    # 给定一个label，把它不可能go to的label输出为一个list，比如 B-body的不可能go to的label有  ['[PAD]','O']等等
    if label.startswith('B') or label.startswith('I'): #B-entity后只可能接I-entity,E-entity
        entity = label.split('-')[-1]
        possible_labels = ['I-'+entity, 'E-'+entity]   #能去的labels
        never_go_lst = []
        for e in tag2idx.keys():
            if e not in possible_labels:
                never_go_lst.append(e)
    elif label.startswith('E') or label.startswith('S') or label=='O':  #E/S后面，不可能出现 I、E
        entity = label.split('-')[-1]
        never_go_lst = []
        for e in tag2idx.keys():
            if e.startswith('I') or e.startswith('E'):
                never_go_lst.append(e)
    else:
        never_go_lst = []
    return never_go_lst


def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0]+torch.log(torch.exp(log_M-torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


def label_sentence_based_terminology(textlist, terminology_dicts):
    ''' Label the sentences based on terminology_dicts [BiMM method]
    
        Args:
            textlist: a list of string. one sentence [w1,w2,w3]
            terminology_dicts: eg. {'class1':set1}
            
        Returns:
            the labels of the sentence based on the terminology dicts
    '''
    dic_union = set()
    for dic_class_name, dic_set in terminology_dicts.items():
        dic_union = dic_union| dic_set
    
    if len(dic_union) == 0:  
        sent_cutted = list(''.join(textlist))
    else:
        bimm_model = BiMM(dic_union, 'union')
        sent_str = ''.join(textlist)
        sent_cutted = bimm_model.cut(sent_str)
       
    sent_label = []
    for w in sent_cutted:
        if w not in dic_union:
            sent_label.append('O')
        else:  # 看w是手术还是药物术语
            for dic_class_name, dic_set in terminology_dicts.items():
                if w in dic_set:
                    if len(w)==1:
                        sent_label.append('S-{}'.format(dic_class_name))
                    else:
                        sent_label += ['B-{}'.format(dic_class_name)] +\
                                     ['I-{}'.format(dic_class_name)]*int(len(w)-2)+\
                                     ['E-{}'.format(dic_class_name)]
    return sent_label 


def plot_fig(lst, num_epoch, out_file):
    ''' Plot the values in list versus epoch_id
    
    Args:
        lst: a list of evaluation values
        num_epoch: total epoch num
        out_file: file name
    Returns:
        a figure
    '''
    
    plt.figure()
    plt.title('Validation Loss vs. Number of Training Epochs')
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Loss')
    plt.plot(range(1, num_epoch + 1), lst)
    plt.xticks(np.arange(1, num_epoch + 1, 1.0))
    plt.savefig(out_file)   

def boolean_string(s):
    ''' Check s string is true or false.
    Args:
        s: the string
    Returns:
        boolean
    '''
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
