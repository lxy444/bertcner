# coding:utf-8

from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import logging
import os
import random
import json
import codecs
import time

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig, BertLayerNorm, 
                                              BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, qids, text, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid  # 样本句子的 ID
        self.qids = qids  # 每个字的原始位置 ID 
        self.text = text  # 单个字
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids  # 字的 ID
        self.input_mask = input_mask  # padding 为 0, 其他为 1 
        self.segment_ids = segment_ids  # 
        self.label_id = label_id  # 字标签的 ID

def readfile(input_file):
    """Reads a BIO data."""
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        data = []
        words = []
        qids = []
        labels = []
        for line in f:
            contents = line.strip()
            tokens = contents.split(' ')
            if len(tokens) == 3:
                # word = line.strip().split(' ')[0]
                # qid = line.strip().split(' ')[1]
                # label = line.strip().split(' ')[-1]
                word, qid, label = line.strip().split(' ')
            else:
                if len(contents) == 0:
                    assert len(words) == len(labels)
                    data.append((words, qids, labels))
                    words = []
                    qids = []
                    labels = []
                    continue
            if contents.startswith("-DOCSTART-"):
                words.append('')
                continue
            words.append(word)
            qids.append(qid)
            labels.append(label)

        return data


def read_conll2003(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_label_list(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    # def __init__(self):
    #     self._label_list = ['[PAD]', 'O', 'B-anatomy', 'I-anatomy', 'E-anatomy', 'S-anatomy', 
    #                    'B-symptom_description', 'I-symptom_description', 'E-symptom_description', 'S-symptom_description', 
    #                    'B-independent_symptom', 'I-independent_symptom', 'E-independent_symptom', 'S-independent_symptom', 
    #                    'B-medicine', 'I-medicine', 'E-medicine', 'S-medicine', 
    #                    'B-surgery', 'I-surgery', 'E-surgery', 'S-surgery', '[CLS]', '[SEP]', 'X']
    #     self._num_labels = len(self._label_list)
    #     self._tag2idx = {label: i for i, label in enumerate(self._label_list, 0)}
    #     self._idx2tag = dict([val, key] for key, val in self._tag2idx.items())


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def _create_examples(self,lines,set_type):
        examples = []
        for i, (sentence, qids, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = ' '.join(sentence)
            qids = qids
            label = label
            examples.append(InputExample(guid=guid, qids=qids, text=text, label=label))
        return examples


def convert_examples_to_features(examples, tag2idx, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    
    ## 此处的 tokenizer 为 bert 里的 tokenizer, 获取每个字的 ID
    ## tag2idx 获取每个标签的 ID
    
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        
        # print(textlist)
        # print(tokens)
        # print(labels)
        
        ## 超出最大长度进行截断
        if len(tokens) >= max_seq_length - 1:
            print('Example No.{} is too long, length is {}, truncated to {}!'.format(ex_index, len(tokens), max_seq_length))
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        
        ntokens = []
        segment_ids = []
        label_ids = []
        
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(tag2idx["[CLS]"])
        
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(tag2idx[labels[i]])
        
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(tag2idx["[SEP]"])
        
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  ## 利用 bert 得到每个字的 ID
        
        input_mask = [1] * len(input_ids)
        
        ## 小于最大长度用 0 补齐, 在字典中 tokenizer.convert_ids_to_tokens([0]) 就是 [PAD]
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)    # 补齐单词的标签设置为 0
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features


def get_pytorch_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)    
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    
    
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return data





# ################################################# 调试数据类
    
# data_dir = './data/ccks2018'
# bert_model = '/Users/lixiangyang/Documents/nlp/bert/model/bert-base-chinese-pytorch'
# max_seq_length = 500
# batch_size = 6


# ## data = read_conll2003('./data/conll2003/train.txt')
# ## print(data[0])

# processor = NerProcessor()

# lines = readfile('./data/ccks2018/train.txt')
# print(lines[0])

# train_examples = processor.get_train_examples(data_dir)
# print(train_examples[0].guid)
# print(train_examples[0].text)
# print(train_examples[0].label)


# label_list = processor.get_label_list()
# tag2idx = processor.get_tag2idx()
# num_labels = len(label_list) + 1    # 为什么要加一, 将 [PAD] 标签设置为 0

# tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case = False)

# ## 注意进去调试 convert_examples_to_features()
# examples = train_examples
# example = examples[0]

# train_features = convert_examples_to_features(
#     train_examples, label_list, max_seq_length, tokenizer)

# print(train_features[0].input_ids)


# all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
# all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
# all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
# all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


# train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# print(train_data[0])

# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# data_iter = iter(train_dataloader)
# next(data_iter)

# input_ids, input_mask, segment_ids, label_ids = next(data_iter)
# print(input_ids.shape)  # 8 * 500, mini-batch


# model = BertForTokenClassification.from_pretrained(bert_model, num_labels = num_labels)

# loss = model(input_ids, segment_ids, input_mask, label_ids)
# print(loss)


# ##### Let's see how to use BertModel to get hidden states
# # Load pre-trained model (weights)
# emb_model = BertModel.from_pretrained('/Users/lixiangyang/Documents/nlp/bert/model/bert-base-chinese-pytorch')
# emb_model.eval()


# # Predict hidden states features for each layer
# encoded_layers, _ = emb_model(input_ids, segment_ids)

# print(len(encoded_layers))

# print(encoded_layers[-1].shape)





# # ################################################# 调试模型

#### CRF

def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0]+torch.log(torch.exp(log_M-torch.max(log_M, axis)[0][:, None]).sum(axis))

def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


## 修改 logits 
class BERT_CRF_NER(nn.Module):

    def __init__(self, bert_model, tag2idx, max_seq_length, hidden_dim, device):
        super(BERT_CRF_NER, self).__init__()
        self.embedding_dim = 768
        self.hidden_dim = hidden_dim
        self.start_label_id = tag2idx['[CLS]']
        self.stop_label_id = tag2idx['[SEP]']
        self.num_labels = len(tag2idx)
        self.max_seq_length = max_seq_length
        # self.batch_size = batch_size
        self.device=device

        # use pretrainded BertModel 
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the bert into label space.
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2, num_layers = 1, bidirectional = True, batch_first = True)
        self.hidden2label = nn.Linear(self.hidden_dim, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.stop_label_id] = -10000

        # 由于看到出现了B-surgery O I-surgery这样的例子，因此把一些不可能的transfer直接类似于上面这样，设置为-10000
        def find_the_label_never_go(label, tag2idx):
            '''给定一个label，把它不可能go to的label输出为一个list，
                比如 B-body的不可能go to的label有  ['[PAD]','O']等等
            '''
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
        
        for label_i in tag2idx.keys():
            never_go_lst = find_the_label_never_go(label_i,tag2idx)
            if never_go_lst:
                for never_go_to_ele in never_go_lst:
                    self.transitions.data[tag2idx[never_go_to_ele],tag2idx[label_i]] = -10000
        # -------------------------------------------------------------------#
        
        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]
        
        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0
        
        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    ## 此处修改 logits
    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        '''
        sentences -> word embedding -> MLP -> feats
        '''
        batch_size = input_ids.shape[0]

        hidden = (torch.randn(2, batch_size, self.hidden_dim // 2, device = self.device),
                 torch.randn(2, batch_size, self.hidden_dim // 2, device = self.device))

        ## bert_seq_out 即为 embedding
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        
        lstm_out, _ = self.lstm(bert_seq_out, hidden)
        
        bert_feats = self.hidden2label(lstm_out)
        
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        ''' 
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        
        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids



# max_seq_length = 500
# batch_size = 8

# output_dir = './output/'
# bert_model_scale = '/Users/lixiangyang/Documents/nlp/bert/model/bert-base-chinese-pytorch'


# total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# bert_model = BertModel.from_pretrained(bert_model_scale)

# model = BERT_CRF_NER(bert_model, tag2idx, len(label_list), max_seq_length, batch_size, device)

# input_ids, input_mask, segment_ids, label_ids = next(data_iter)

# neg_loss = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)
# print(neg_loss)

# ### 具体查看 CRF 层
# bert_seq_out, _ = model.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
# print(bert_seq_out.shape)

# hidden = model.init_hidden()

# lstm_out, hidden = model.lstm(bert_seq_out, hidden)

# bert_feats = model.hidden2label(lstm_out)


# bert_seq_out = model.dropout(bert_seq_out)

# bert_feats = model.hidden2label(bert_seq_out)
# print(bert_feats.shape)







# #################################################


parser = argparse.ArgumentParser()

## Required parameters

parser.add_argument("--data_dir",
                    default='data/ccks2018/',
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

parser.add_argument("--bert_model", default='/Users/lixiangyang/Documents/nlp/bert/model/bert-base-chinese-pytorch', 
                    type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")


parser.add_argument("--output_dir",
                    default='./output',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")


## Other parameters
parser.add_argument("--cache_dir",
                    default="",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length",
                    default=480,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")

parser.add_argument("--train_batch_size",
                    default=6,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=4,
                    type=int,
                    help="Total batch size for eval.")

parser.add_argument("--hidden_dim",
                    default=128,
                    type=int,
                    help="LSTM hidden dim.")


parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=15,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

args = parser.parse_args()

print(args)

start_time = time.time()


if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

logger.info("device: {} n_gpu: {}, distributed training: {}".format(
    device, n_gpu, bool(args.local_rank != -1)))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                        args.gradient_accumulation_steps))

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


processor = NerProcessor()

# label_list = processor.get_label_list()

if args.data_dir == 'data/ccks2018/':
    label_list = ['[PAD]', 'O', 'B-anatomy', 'I-anatomy', 'E-anatomy', 'S-anatomy', 
                   'B-symptom_description', 'I-symptom_description', 'E-symptom_description', 'S-symptom_description', 
                   'B-independent_symptom', 'I-independent_symptom', 'E-independent_symptom', 'S-independent_symptom', 
                   'B-medicine', 'I-medicine', 'E-medicine', 'S-medicine', 
                   'B-surgery', 'I-surgery', 'E-surgery', 'S-surgery', '[CLS]', '[SEP]', 'X']


if args.data_dir == 'data/ccks2017/':
    label_list = ['[PAD]', 'O', 'B-symptom', 'I-symptom', 'E-symptom', 'S-symptom', 
                   'B-check', 'I-check', 'E-check', 'S-check', 
                   'B-disease', 'I-disease', 'E-disease', 'S-disease', 
                   'B-body', 'I-body', 'E-body', 'S-body', 
                   'B-treatment', 'I-treatment', 'E-treatment', 'S-treatment',
                   'X', '[CLS]', '[SEP]']


tag2idx = {label : i for i, label in enumerate(label_list, 0)}
idx2tag = dict([val, key] for key, val in tag2idx.items())


tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)


## prepare data

## ccks 2018
if args.data_dir == 'data/ccks2018/':

    train_all = processor.get_train_examples(args.data_dir)
    # train_all.extend(train_all[:14])  # 添加 14 个样本, 便于 mini-batch
    # num_dev = round(0.2 * len(train_all))
    num_dev = 504
    dev_indices = np.random.choice(len(train_all), num_dev, replace = False)
    train_indices = [j for j in range(len(train_all)) if j not in dev_indices]
    dev_examples = [train_all[i] for i in dev_indices]
    # train_examples = [train_all[i] for i in train_indices]
    train_examples = train_all  # 使用全部训练数据


    logger.info("number of train examples: {}".format(len(train_examples)))
    logger.info("number of dev examples: {}".format(len(dev_examples)))

    train_features = convert_examples_to_features(train_examples, tag2idx, args.max_seq_length, tokenizer)
    train_data = get_pytorch_dataset(train_features)


    dev_features = convert_examples_to_features(dev_examples, tag2idx, args.max_seq_length, tokenizer)
    dev_data = get_pytorch_dataset(dev_features)


    test_examples = processor.get_test_examples(args.data_dir)
    logger.info("number of test examples: {}".format(len(test_examples)))
    test_features = convert_examples_to_features(test_examples, tag2idx, args.max_seq_length, tokenizer)
    test_data = get_pytorch_dataset(test_features)


## ccks2017
if args.data_dir == 'data/ccks2017/':

    train_all = processor.get_train_examples(args.data_dir)
    # train_all.extend(train_all[:6])  # 添加 6 个样本, 便于 mini-batch
    num_dev = round(0.2 * len(train_all))
    # num_dev = 976
    dev_indices = np.random.choice(len(train_all), num_dev, replace = False)
    train_indices = [j for j in range(len(train_all)) if j not in dev_indices]
    dev_examples = [train_all[i] for i in dev_indices]
    # train_examples = [train_all[i] for i in train_indices]
    train_examples = train_all  # 使用全部训练数据


    logger.info("number of train examples: {}".format(len(train_examples)))
    logger.info("number of dev examples: {}".format(len(dev_examples)))

    train_features = convert_examples_to_features(train_examples, tag2idx, args.max_seq_length, tokenizer)
    train_data = get_pytorch_dataset(train_features)


    dev_features = convert_examples_to_features(dev_examples, tag2idx, args.max_seq_length, tokenizer)
    dev_data = get_pytorch_dataset(dev_features)


    test_examples = processor.get_test_examples(args.data_dir)
    logger.info("number of test examples: {}".format(len(test_examples)))
    test_features = convert_examples_to_features(test_examples, tag2idx, args.max_seq_length, tokenizer)
    test_data = get_pytorch_dataset(test_features)



# all_train_len = [len(item.label) for item in train_examples]
# all_test_len = [len(item.label) for item in test_examples]


num_train_optimization_steps = int(
    len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

if args.local_rank != -1:
    num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


## 构建模型
# cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

# model = BertForTokenClassification.from_pretrained(args.bert_model,
#           cache_dir=cache_dir,
#           num_labels = num_labels)



bert_model = BertModel.from_pretrained(args.bert_model)

# if args.do_train:
#     model = BERT_CRF_NER(bert_model, tag2idx, args.max_seq_length, args.train_batch_size, args.hidden_dim, device)

# if args.do_eval:
#     model = BERT_CRF_NER(bert_model, tag2idx, args.max_seq_length, args.eval_batch_size, args.hidden_dim, device)

model = BERT_CRF_NER(bert_model, tag2idx, args.max_seq_length, args.hidden_dim, device)

if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)


model.to(device)

## 构建优化器
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0



## 想打印出每轮训练结束后的准确率
def evaluate(model, eval_data, out_file):
    
    # Run prediction for full data

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    model.eval()


    y_true = []
    y_pred = []
    raw_text = []

    tag_res = open(out_file, 'w', encoding = 'utf-8')
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        with torch.no_grad():
            _, predicted_ids = model(input_ids, segment_ids, input_mask)    # bert + crf
        
        predicted_ids = predicted_ids.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        
        for i,mask in enumerate(input_mask):
            x = input_ids[i]
            x_words = tokenizer.convert_ids_to_tokens(x.tolist())
            
            temp_1 =  []
            temp_2 = []
            temp_word = []
            
            for j,m in enumerate(mask):
                if j == 0:
                    continue  # 句子开始 [CLS]
                if m and idx2tag[label_ids[i][j]] != "X" and idx2tag[label_ids[i][j]] != '[SEP]':
                    temp_1.append(idx2tag[label_ids[i][j]])
                    temp_2.append(idx2tag[predicted_ids[i][j]])
                    temp_word.append(x_words[j])
                    line = x_words[j] + ' ' + idx2tag[label_ids[i][j]] + ' ' + idx2tag[predicted_ids[i][j]]
                    tag_res.write(line + '\n')
                else:
                    temp_1.pop()
                    temp_2.pop()
                    temp_word.pop()
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    raw_text.append(temp_word)
                    line = ''
                    break
            tag_res.write('\n')
    
    tag_res.close()

    
    report = classification_report(y_true, y_pred, digits = 4)

    acc = (precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))

    model.train()

    return acc, report







if args.do_train:

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    
    
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()

    f1_prev = 0
    acc_history = []
    loss_history = []

    for num_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # loss = model.module.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)
            loss = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)
            
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
        out_file = 'epoch_' + str(num_epoch) + '_tag.txt'
        out_file = os.path.join(args.output_dir, out_file) 
        (p_score, r_score, f_score), _ = evaluate(model, test_data, out_file)

        loss_history.append(loss)
        acc_history.append(f_score)

        logger.info(("Epoch: {}, precision: {}, recall: {}, f1 score: {}".format(num_epoch, p_score, r_score, f_score)))

        # if f_score > f1_prev:
   
    torch.save({'epoch': num_epoch, 'model_state': model.state_dict(), 
            'max_seq_length': args.max_seq_length},
            os.path.join(args.output_dir, 'model.ckpt'))


            # f1_prev = f_score
    
    ## plot curve
    l_hist = []
    l_hist = [h.cpu().detach().numpy() for h in loss_history]

    ## loss 曲线
    plt.figure(1)
    plt.title('Validation Loss vs. Number of Training Epochs')
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Loss')
    plt.plot(range(1, args.num_train_epochs + 1), l_hist)
    plt.xticks(np.arange(1, args.num_train_epochs + 1, 1.0))
    plt.savefig(os.path.join(args.output_dir, 'loss.pdf'))

    ## 准确率曲线
    plt.figure(2)
    plt.title('Validation Accuracy vs. Number of Training Epochs')
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Accuracy')
    plt.plot(range(1, args.num_train_epochs + 1), acc_history)
    plt.xticks(np.arange(1, args.num_train_epochs + 1, 1.0))
    plt.savefig(os.path.join(args.output_dir, 'acc.pdf'))



# Load a trained model and config that you have fine-tuned
else:
    checkpoint = torch.load(args.output_dir + '/model.ckpt', map_location = device)
    epoch = checkpoint['epoch']
    pretrained_dict = checkpoint['model_state']
    
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)


model.to(device)





if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

    logger.info("***** Running evaluating *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_data = test_data
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    model.eval()


    y_true = []
    y_pred = []
    raw_text = []
   
    tag_res = open(os.path.join(args.output_dir, 'tag_result.txt'), 'w', encoding = 'utf-8')
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        # raw_words = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        # print(raw_words)
        
        # raw_labels = label_ids[0]
        # raw_labels = raw_labels[raw_labels > 0]
        # raw_labels = [idx2tag[i] for i in raw_labels.tolist()]
        

        with torch.no_grad():
            _, predicted_ids = model(input_ids, segment_ids, input_mask)    # bert + crf
        
        predicted_ids = predicted_ids.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        
        for i,mask in enumerate(input_mask):
            x = input_ids[i]
            x_words = tokenizer.convert_ids_to_tokens(x.tolist())
            
            temp_1 =  []
            temp_2 = []
            temp_word = []
            
            for j,m in enumerate(mask):
                if j == 0:
                    continue  # 句子开始 [CLS]
                if m and idx2tag[label_ids[i][j]] != "X" and idx2tag[label_ids[i][j]] != '[SEP]':
                    temp_1.append(idx2tag[label_ids[i][j]])
                    temp_2.append(idx2tag[predicted_ids[i][j]])
                    temp_word.append(x_words[j])
                    line = x_words[j] + ' ' + idx2tag[label_ids[i][j]] + ' ' + idx2tag[predicted_ids[i][j]]
                    tag_res.write(line + '\n')
                else:
                    temp_1.pop()
                    temp_2.pop()
                    temp_word.pop()
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    raw_text.append(temp_word)
                    line = ''
                    break
            tag_res.write('\n')
    
    tag_res.close()
    
    report = classification_report(y_true, y_pred, digits = 4)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)



end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))




