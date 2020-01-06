# -*- coding: utf-8 -*-
"""
NER model: BERT embedding + BiLSTM + CRF + (radical feature) + (dictionary feature post-processing)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from data_reader.utils import find_the_label_never_go, log_sum_exp_batch, one_hot


class NER_Model(nn.Module):

    def __init__(self, bert_model, tag2idx, max_seq_length, 
                 device, tokenizer, terminology_dicts, constant,
                 lstm_hidden_num, lstm_bidirectional=True, lstm_layer=1,
                 add_radical_or_not=False, radical2idx=dict(), 
                 radical_one_hot=False, radical_emb_dim=20,
                 dropout=0.2, embedding_dim=768):
        """
        Args:
            bert_model: the BERT model
            tag2idx: {tag: id}
            max_seq_length: max sequence length
            device: gpu or cpu
            tokenier: the tokenizer
            terminologi_dicts: eg. {'medicine':set1, 'surgery':set2}
            constant: the constant which is added to the logits
            lstm_hidden_num: the hidden units num of LSTM
            lstm_bidirectional: use LSTM or BiLSTM. Default is True.
            lstm_layer: the number of LSTM layer. Default is 1.
            add_radical_or_not: whether radical feature is used or not. Default is False.
            radical2idx: {radical: id}
            radical_one_hot: If radical feature is used, radical is mapped into one-hot encoding or randomly initialized. Defualt is False.
            radical_emb_dim: If radical feature is used and it is randomly initialized, the dim of the embedding is set. Default is 20.
            dropout: the drop out prob
            embedding_dim: the dim of embedding. Default is 768.
        
        Raise Error:
            If the add_radical_or_not is True, the radical_dict_path does not exist.
        """
        super(NER_Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_num = lstm_hidden_num
        self.start_label_id = tag2idx['[CLS]']
        self.stop_label_id = tag2idx['[SEP]']
        self.num_labels = len(tag2idx)
        self.max_seq_length = max_seq_length
        self.device = device
        self.tokenizer = tokenizer
        self.terminology_dicts = terminology_dicts
        self.tag2idx = tag2idx
        self.idx2tag = dict([val, key] for key, val in tag2idx.items())
        self.constant = constant  
        self.add_radical_or_not = add_radical_or_not
        self.radical_one_hot = radical_one_hot
        self.radical2idx = radical2idx
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_layer = lstm_layer
        self.radical_emb_dim = radical_emb_dim
        self.label_id_interested = np.array(list(map(lambda s: s[0] in set('BIES'), list(tag2idx.keys())))).astype('float32') # 也就是只有这些为1对应的label才对应乘以一个constant用来加到logits上
        
        # load radical dict
        if self.add_radical_or_not:
            self.radical_num = len(self.radical2idx)
        
        # LSTM
        if self.add_radical_or_not and not self.radical_one_hot:  # radical random embedding with the char emb as the input of LSTM
            self.radical_embeddings = nn.Embedding(self.radical_num + 1, self.radical_emb_dim)  # edit: + 1
            self.lstm = nn.LSTM(self.embedding_dim + self.radical_emb_dim, 
                                lstm_hidden_num, 
                                num_layers = self.lstm_layer, 
                                bidirectional = lstm_bidirectional, 
                                batch_first = True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, 
                                lstm_hidden_num, 
                                num_layers = self.lstm_layer,
                                bidirectional = lstm_bidirectional, 
                                batch_first = True)
   
        if lstm_bidirectional:
            self.hidden_dim = int(lstm_hidden_num*2)
        else:
            self.hidden_dim = lstm_hidden_num
        
        # linear layer
        if self.add_radical_or_not and self.radical_one_hot:
            self.hidden2label = nn.Linear(self.hidden_dim+self.radical_num, self.num_labels)
        else:
            self.hidden2label = nn.Linear(self.hidden_dim, self.num_labels)
        
        # Matrix of transition parameters.  
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels)) # Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions.data[self.start_label_id, :] = -10000 # never transfer *to* start
        self.transitions.data[:, self.stop_label_id] = -10000 # never transfer *from* stop
        
        for label_i in tag2idx.keys():
            never_go_lst = find_the_label_never_go(label_i,tag2idx)  # 把一些不可能的transfer直接类似于上面这样，设置为-10000
            if never_go_lst:
                for never_go_to_ele in never_go_lst:
                    self.transitions.data[tag2idx[never_go_to_ele],tag2idx[label_i]] = -10000
        
        # initialize
        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''
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


    def _get_radical_emb(self, radical_ids):
        ''' Get the radical embeddings
        '''
        batch_size = radical_ids.shape[0]
        embeds = self.radical_embeddings(radical_ids).view(batch_size, self.max_seq_length, -1)
        return embeds
    
    
    def _get_radical_one_hot(self, radical_ids):
        ''' Get the radical one-hot encoding  
        
        Args:
            radical_ids: the radical ids for each token. shape(batch_size, max_seq_len)
        
        Returns:
            the one-hot encoding. shape (batch_size, max_seq_len, radical_num)
        '''
        radical_one_hot = one_hot(radical_ids, self.radical_num)
        if torch.cuda.is_available():
            radical_one_hot_tensor = torch.FloatTensor(radical_one_hot).cuda()
        else:
            radical_one_hot_tensor = torch.FloatTensor(radical_one_hot)
        return radical_one_hot_tensor
    
    def _add_constant_to_logits(self, feats, label_ids_based_terminology):
        ''' Add constant based on the terminolgoy dicts.
        
        Args:
            feats: the logits (i.e. the predicted prob for each label in each token).
                The shape is (batch_size, max_seq_len, num_labels)
            label_ids_based_terminology: the predicted label ids based on terminology [BiMM method]. 
                The shape is (batch_size, max_seq_length)
        
        Returns:
            the modified logits
        '''
        constant_mask_array = torch.from_numpy(self.label_id_interested).repeat(self.max_seq_length,1)  #shape (max_seq_len, num_labels)
        one_hot_based_terminology = one_hot(label_ids_based_terminology, len(self.tag2idx)) #shape: (batch_size, max_seq_len, num_labels)
        constant_array = self.constant * constant_mask_array*one_hot_based_terminology  #shape: (batch_size, max_seq_len, num_labels)
        
        ## edit:
        if torch.cuda.is_available():
            constant_array = constant_array.cuda()

        feats_modify = feats + constant_array
        return feats_modify


    def _get_bert_features(self, input_ids, segment_ids, input_mask, 
                           radical_ids, label_ids_based_terminology):
        '''
        sentences -> word embedding -> MLP -> feats
        
        Args:
            input_ids: token id
            segment_ids: segment id
            input_mask: mask
            radical_ids: radical ids
            label_ids_based_terminolgoy: the label ids based on terminology [BiMM method]
        Returns:
            features
        '''
        batch_size = input_ids.shape[0]
       
        if self.lstm_bidirectional:
            hidden = (torch.randn(2, batch_size, self.lstm_hidden_num, device = self.device),
                     torch.randn(2, batch_size, self.lstm_hidden_num, device = self.device))
        else:
            hidden = torch.randn(2, batch_size, self.lstm_hidden_num, device = self.device)
        
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False) #shape:[batch_size, max_len, 768]
        
        # radical feature 
        if self.add_radical_or_not and not self.radical_one_hot:
            radical_emb = self._get_radical_emb(radical_ids)  
            lstm_input = torch.cat((bert_seq_out, radical_emb), dim = -1)
        else:
            lstm_input = bert_seq_out
            
        lstm_out, _ = self.lstm(lstm_input, hidden) # shape:[batch_size, max_len, hidden_dim]
        
        if self.add_radical_or_not and self.radical_one_hot:
            radical_one_hot =  self._get_radical_one_hot(radical_ids)
            concat_fea = torch.cat((lstm_out, radical_one_hot),dim=2)
            feats = self.hidden2label(concat_fea)
        else:
            feats = self.hidden2label(lstm_out) # shape:[batch_size, max_len, num_labels]
        
        # modify feats based on the terminology dicts [修改logits，即基于双向最大匹配结果，把药物、手术对应的label概率加一个常数]
        if self.constant == 0:
            feats_modified = feats
        else:
            feats_modified = self._add_constant_to_logits(feats, label_ids_based_terminology)
        return feats_modified
    

    def _score_sentence(self, feats, label_ids):
        ''' 
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(self.device)
        # the 0th node is start_label->start_word, the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score
    

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]
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


    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, radical_ids, label_ids_based_terminology, label_ids):  
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask, radical_ids, label_ids_based_terminology)
        forward_score = self._forward_alg(bert_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    
    def forward(self, input_ids, segment_ids, input_mask, radical_ids, label_ids_based_terminology):
        """ Forward function just for prediction.
        
            Args:
                 input_ids: token id
                 segment_ids: segment id
                 input_mask: mask
                 radical_ids: radical ids
                 label_ids_based_terminolgoy: the label ids based on terminology [BiMM method]
                
            Returns:
                score, label_pred
        """
        # this forward is just for prediction, not for training. dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask, radical_ids, label_ids_based_terminology)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids

    
    def _print_model_information(self):
        """ Print the model information
            
            Returns:
                information about the model parameters
        """
        string = f'device={self.device}, max_seq_length={self.max_seq_length}, constant={self.constant}, '\
                f'lstm: hidden_num={self.lstm_hidden_num},'\
                f'bidirectional={self.lstm_bidirectional}, lstm_layer={self.lstm_layer}\n'\
                f'radical: add_radical={self.add_radical_or_not}, radical_one_hot={self.radical_one_hot},'\
                f'radical_emb_dim={self.radical_emb_dim},\n radical2idx:{self.radical2idx}\n'\
                f'tag2idx:{self.tag2idx}\n'
        return string
    
    
    def _evaluate_on_eval_data(self, eval_data, out_file, eval_batch_size):
        '''Evaluate the model on eval_data and print the predicted result in out_file.
        
        Args:
            eval_data: evaluation dataset
            out_file: the file name
            eval_batch_size: batch size during the evaluation
        
        Returns:
            the out_file which contains the predicted labels for each token
        '''
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        self.eval()  # Set the module in evaluation mode.
        
        y_true = []
        y_pred = []
        raw_text = []

        tag_res = open(out_file, 'w', encoding = 'utf-8')

        for input_ids, input_mask, segment_ids, radical_ids, label_ids_based_terminology, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            radical_ids = radical_ids.to(self.device)
            label_ids_based_terminology = label_ids_based_terminology.to(self.device)
            label_ids = label_ids.to(self.device)
    
            with torch.no_grad():
                _, predicted_ids = self.forward(input_ids, segment_ids, input_mask, radical_ids, label_ids_based_terminology)   
            
            predicted_ids = predicted_ids.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            
            for i,mask in enumerate(input_mask):
                x = input_ids[i]
                x_words = self.tokenizer.convert_ids_to_tokens(x.tolist())
    
                temp_1 =  []
                temp_2 = []
                temp_word = []
                
                for j,m in enumerate(mask):
                    if j == 0:
                        continue 
                    if m and self.idx2tag[label_ids[i][j]] != "X" and self.idx2tag[label_ids[i][j]] != '[SEP]':
                        temp_1.append(self.idx2tag[label_ids[i][j]])
                        temp_2.append(self.idx2tag[predicted_ids[i][j]])
                        temp_word.append(x_words[j])
                        line = x_words[j] + ' ' + self.idx2tag[label_ids[i][j]] + ' ' + self.idx2tag[predicted_ids[i][j]]
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
        acc = (precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
        report = classification_report(y_true, y_pred, digits = 4)
        self.train()
        return acc, report
