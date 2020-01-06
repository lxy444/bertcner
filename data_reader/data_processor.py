# -*- coding: utf-8 -*-
import codecs
import os
import torch
from torch.utils.data import TensorDataset
import sys
sys.path.append('..')
from data_reader.utils import label_sentence_based_terminology, padding


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
    

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, qids, text, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid  # 样本句子的 ID
        self.qids = qids  # 每个字的原始位置 ID 
        self.text = text  # 单个字
        self.label = label
        
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, radical_ids,
                 label_id_based_terminology):
        '''
        Args:
            input_ids: token id
            input_mask: mask
            segment_ids
            label_id: true label id for each token
            radical_ids: radical id for each token
            label_id_based_terminology; predicted label id via terminology
        '''
        self.input_ids = input_ids 
        self.input_mask = input_mask  # padding 为 0, 其他为 1 
        self.segment_ids = segment_ids  
        self.label_id = label_id  
        self.radical_ids = radical_ids
        self.label_id_based_terminology = label_id_based_terminology
        
        
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
    

def convert_examples_to_features(examples, tag2idx, max_seq_length, tokenizer,
                                 radical_dict, radical2idx, terminology_dicts):
    '''Loads a data file into a list of `InputBatch`s.
        
    Args:
        examples: class examples
        tag2idx: {tag: idx}
        max_seq_length: max sequence length
        tokenizer: Tokenizer class
        radical_dict: {char: radical} dict
        radical2idx: {radical: idx}
        terminology_dicts: eg. {'class1':set1,'class2':set2}
        
    Returns:
        Feature class contain input_ids, input_mask, segment_ids, label_id, 
        radical_ids, label_id_based_terminology
    '''
    
    # 此处的 tokenizer 为 bert 里的 tokenizer, 获取每个字的 ID
    # tag2idx 获取每个标签的 ID
    
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text.split(' ')
        labellist = example.label   # true labels
        labellist_based_terminology =  label_sentence_based_terminology(textlist, terminology_dicts)  # labels predicted via terminology_dicts
        
        tokens = []
        labels = []
        labels_based_terminology = []
        radicals = []  
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            label_1_based_terminology = labellist_based_terminology[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    labels_based_terminology.append(label_1_based_terminology)
                else:
                    labels.append("X")
                    labels_based_terminology.append("X")
            for t in token:
                radicals.append(radical_dict.get(t))
            
        # 超出最大长度进行截断
        if len(tokens) >= max_seq_length - 1:
            print('Example No.{} is too long, length is {}, truncated to {}!'.format(ex_index, len(tokens), max_seq_length))
            tokens = tokens[0 : (max_seq_length - 2)]
            labels = labels[0 : (max_seq_length - 2)]
            labels_based_terminology = labels_based_terminology[0 : (max_seq_length - 2)]
            radicals = radicals[0 : (max_seq_length - 2)]
            
        ntokens = ["[CLS]"]
        segment_ids = [0]
        label_ids = [tag2idx["[CLS]"]]
        label_ids_based_terminology = [tag2idx["[CLS]"]]
        radical_ids = [radical2idx["[CLS]"]]
        
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(tag2idx[labels[i]])
            label_ids_based_terminology.append(tag2idx[labels_based_terminology[i]])
            radical_ids.append(radical2idx[radicals[i]])
        
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(tag2idx["[SEP]"])
        label_ids_based_terminology.append(tag2idx["[SEP]"])
        radical_ids.append(radical2idx["[SEP]"])
        
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # token_ids
        input_mask = [1] * len(input_ids)
        
        # padding
        input_ids = padding(input_ids, max_seq_length, 0)
        input_mask = padding(input_mask, max_seq_length, 0)
        segment_ids = padding(segment_ids, max_seq_length, 0)
        label_ids = padding(label_ids, max_seq_length, tag2idx['[PAD]'])
        label_ids_based_terminology = padding(label_ids_based_terminology, max_seq_length, tag2idx['[PAD]'])
        radical_ids = padding(radical_ids, max_seq_length, radical2idx['[PAD]'])
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              radical_ids=radical_ids,
                              label_id_based_terminology=label_ids_based_terminology))
    return features


def get_pytorch_dataset(features):
    '''Transform the features into torch.tensor
    
    Args:
        features: list of InputFeatures
    Returns:
        torch tensor dataset consisting of input_ids, input_mask, segment_ids,
        radical_ids, label_ids_based_terminology, label_ids 
    '''
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)    
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_radical_ids = torch.tensor([f.radical_ids for f in features], dtype=torch.long)
    all_label_ids_based_terminology = torch.tensor([f.label_id_based_terminology 
                                                    for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, 
                         all_radical_ids, all_label_ids_based_terminology,
                         all_label_ids)
    return data
