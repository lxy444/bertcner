# -*- coding: utf-8 -*-
"train and evaluate NER model"

import argparse
from collections import OrderedDict
import torch
import random
import numpy as np
import logging
import os
import time
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm

from model import NER_Model
from data_reader.data_processor import NerProcessor, convert_examples_to_features, get_pytorch_dataset
from data_reader.data_loader import load_label_list, load_radical_dict, load_terminology_dict
from data_reader.utils import plot_fig, boolean_string

# path parameters
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default='data/ccks_2018',
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", 
                    default='bert_model', 
                    type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--output_dir",
                    default='./output',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")


parser.add_argument('--terminology_dicts_path',
                    type=str,
                    help='the path of such termonology dicts as medicine_dict and surgery_dict',
                    default="{'medicine':'data/ccks_2018/drug_dict.txt',  'surgery':'data/ccks_2018/surgery_dict.txt'}")
parser.add_argument('--radical_dict_path',
                   type=str,
                   help='The path of the file which contains "character,radical" in each row',
                   default='data/radical_dict.txt')

# other parameters
parser.add_argument('--constant',
                    type=float,
                    help='the constant which is added to the logits [i.e. the output of BiLSTM]',
                    default=0)
parser.add_argument('--add_radical_or_not',
                    type=boolean_string,
                    help='whether use radical feature or not',
                    default=False)
parser.add_argument('--radical_one_hot',
                    type=boolean_string,
                    help='If radical feature is used, the radical feature is mapped into one-hot encoding or randomly initialized',
                    default=False)
parser.add_argument('--radical_emb_dim',
                    type=int,
                    default=20,
                    help='the dim of randomly initialized embedding for radicals')
parser.add_argument("--cache_dir",
                    default="",
                    type=str,
                    help="Where do you want to store the pre-trained models")
parser.add_argument("--max_seq_length",
                    default=480,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=True,
                    type=boolean_string,
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=True,
                    type=boolean_string,
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
                    default=64,    
                    type=int,
                    help="LSTM hidden dim.")
parser.add_argument('--lstm_bidirectional',
                    default=True,
                    type=boolean_string,
                    help='LSTM is bidirectional or not')
parser.add_argument('--lstm_layer',
                    default=1,
                    type=int,
                    help='LSTM layer num')
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=5,
                    type=int,  
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('--dropout',
                    default=0.2,
                    type=float,
                    help='the dropout prob during training')
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
parser.add_argument('--gpu_id',
                    type=int,
                    default=2,
                    help="the gpu id for training")


if __name__ == '__main__':

    start_time = time.time()

    ############################ Prepare #############################
    # parse args
    args = parser.parse_args()
    
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    
    # load 
    processor = NerProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    bert_model = BertModel.from_pretrained(args.bert_model)

    # label2id
    label_list = load_label_list(os.path.join(args.data_dir, 'label.txt'))
    tag2idx = OrderedDict({label : i for i, label in enumerate(label_list, 0)})

    # radical
    ## edit:
    # radical_dict = joblib.load('data/radical.pkl')
    # radical2idx = dict([val, i] for i, val in enumerate(radical_dict.values(), 1))

    radical_dict, radical_lst = load_radical_dict(args.radical_dict_path) if args.add_radical_or_not else (dict(),[]) # {char: radical}
    radical_lst = ['[PAD]', None, '[CLS]', '[SEP]'] + radical_lst
    radical2idx = OrderedDict({radical: i for i, radical in enumerate(radical_lst, 0)})
    
    
    # terminology_dicts
    terminology_dicts = load_terminology_dict(eval(args.terminology_dicts_path))
    
    # prepare data
    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    logger.info("number of train examples: {}".format(len(train_examples)))
    logger.info("number of dev examples: {}".format(len(dev_examples)))
    logger.info("number of test examples: {}".format(len(test_examples)))

    train_features = convert_examples_to_features(train_examples, tag2idx, 
                                                  args.max_seq_length, tokenizer,
                                                  radical_dict, radical2idx,
                                                  terminology_dicts)
    dev_features = convert_examples_to_features(dev_examples, tag2idx, 
                                                args.max_seq_length, tokenizer,
                                                radical_dict, radical2idx,
                                                terminology_dicts)
    test_features = convert_examples_to_features(test_examples, tag2idx, 
                                                 args.max_seq_length, tokenizer,
                                                 radical_dict, radical2idx,
                                                 terminology_dicts)
    
    train_data = get_pytorch_dataset(train_features)  #input_ids, input_mask, segment_ids, radical_ids, label_ids_based_terminology, label_ids
    dev_data = get_pytorch_dataset(dev_features)   
    test_data = get_pytorch_dataset(test_features)
    
    ##########################  Model  #####################################
    model = NER_Model(bert_model = bert_model, 
                      tag2idx = tag2idx, 
                      max_seq_length = args.max_seq_length, 
                      device = device, 
                      tokenizer = tokenizer,
                      terminology_dicts = terminology_dicts, 
                      constant = args.constant,
                      lstm_hidden_num = args.hidden_dim,
                      lstm_bidirectional = args.lstm_bidirectional,
                      lstm_layer = args.lstm_layer,
                      add_radical_or_not = args.add_radical_or_not,
                      radical2idx = radical2idx,
                      radical_one_hot = args.radical_one_hot,
                      radical_emb_dim = args.radical_emb_dim,
                      dropout = args.dropout,
                      embedding_dim = 768)
    model.to(device)
    logger.info(model._print_model_information())
    
    ########################## Train  ######################################
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
     
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # optimizer
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
    
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        model.constant = 0  # comment: constant is set as 0 during training stage
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
        model.train()
    
        f1_prev = 0
        f1_history = []  
        loss_history = []  
    
        for num_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, radical_ids, label_ids_based_terminology, label_ids = batch
                loss = model.neg_log_likelihood(input_ids, segment_ids, input_mask, radical_ids, label_ids_based_terminology, label_ids)
                
                if n_gpu > 1:
                    loss = loss.mean()  # to average on multi-gpu.
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
            
            # eval on dev_data
            out_file = os.path.join(args.output_dir, 'epoch_' + str(num_epoch) + '_tag.txt') 
            (p_score, r_score, f_score), _ = model._evaluate_on_eval_data(test_data, out_file, args.eval_batch_size)
            loss_history.append(loss)
            f1_history.append(f_score)
            logger.info(("Epoch: {}, precision: {}, recall: {}, f1 score: {}".format(num_epoch, p_score, r_score, f_score)))

  
        torch.save({'epoch': num_epoch, 'model_state': model.state_dict(), 
                'max_seq_length': args.max_seq_length},
                os.path.join(args.output_dir, 'model.ckpt'))
            
            # if f_score >= f1_prev:
            #     f1_prev = f_score
            #     logger.info('save the best model on dev_data: epoch {} model'.format(str(num_epoch)))
            #     torch.save({'epoch': num_epoch, 'model_state': model.state_dict(), 
            #                 'max_seq_length': args.max_seq_length},
            #                  os.path.join(args.output_dir, 'model.ckpt'))
                
        # plot curve
        loss_lst = [h.cpu().detach().numpy() for h in loss_history]
        plot_fig(loss_lst, args.num_train_epochs, os.path.join(args.output_dir, 'dev_loss.pdf'))
        plot_fig(f1_history, args.num_train_epochs, os.path.join(args.output_dir, 'dev_f1.pdf'))    

    else: # Load a trained model
        checkpoint = torch.load(args.output_dir + '/model.ckpt', map_location = device)  
        epoch = checkpoint['epoch']
        pretrained_dict = checkpoint['model_state']
        
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)


    ##########################    Eval      ################################
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("***** Running evaluating *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        logger.info(model._print_model_information())
        (p_score, r_score, f_score), report  = model._evaluate_on_eval_data(test_data, os.path.join(args.output_dir, 'test_tag.txt'), args.eval_batch_size) 
        output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



