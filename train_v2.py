import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer, BertTokenizer, BertModel, RoFormerModel, RoFormerTokenizer
)
from models_v2 import inference_model
from data_loader_v2 import data_loader
from bert_model import BertForSequenceEncoder, Net
from torch.nn import NLLLoss
import logging
import json
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# 计算pcc
def pearson_correlation(pred, ref):
    """ Computes Pearson correlation """
    from scipy.stats import pearsonr
    pc = pearsonr(pred, ref)
    return pc[0]  # return correlation value and ignore p,value


def eval_model(model, validset_reader):
    model.eval()
    pred_list = list()
    golden_list = list()
    with torch.no_grad():
        for step, batch_data in enumerate(validset_reader):
            # for step, (inp_tensor, msk_tensor, seg_tensor, score_tensor) in enumerate(validset_reader):
            inp_tensor = batch_data["input_ids"].to(args.device)
            msk_tensor = batch_data["attention_mask"].to(args.device)
            seg_tensor = batch_data["token_type_ids"].to(args.device)
            lab_tensor = batch_data["labels"].to(args.device)
            # for index, data in enumerate(validset_reader):
            #     inp_tensor, msk_tensor, seg_tensor, lab_tensor = data
            prob = model(inp_tensor, msk_tensor, seg_tensor)
            topic_scores, topic_idxs = torch.topk(prob, k=1, dim=0)
            pred_list.extend(topic_idxs.view(-1).tolist())
            golden_list.extend(lab_tensor.view(-1).tolist())

    acc = accuracy_score(pred_list, golden_list)
    precision = precision_score(pred_list, golden_list, average="micro")
    recall = recall_score(pred_list, golden_list, average="micro")
    micro_f1 = f1_score(pred_list, golden_list, average="micro")

    return acc


def train_model(model, ori_model, args, trainset_reader, validset_reader):
    save_path = args.outdir + '/model'
    best_pcc = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.dataset.__len__() / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    # optimizer = optim.Adam(model.parameters(), args.learning_rate)
    global_step = 0
    crit = nn.CrossEntropyLoss()
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        # for inp_tensor, msk_tensor, seg_tensor, score_tensor in trainset_reader:
        #     model.train()
        for step, batch_data in enumerate(trainset_reader):
            model.train()
            inp_tensor = batch_data["input_ids"].to(args.device)
            msk_tensor = batch_data["attention_mask"].to(args.device)
            seg_tensor = batch_data["token_type_ids"].to(args.device)
            score_tensor = batch_data["labels"].to(args.device)

            score = model(inp_tensor, msk_tensor, seg_tensor)
            #score_tensor = score_tensor.view(-1)
            score = score.view(-1,10)
            score = score.to(torch.float32)
            loss = crit(score, score_tensor)
            running_loss += loss.item()
            if args.gradient_accumulation_steps != 0:
                loss = loss / args.gradient_accumulation_steps
            # loss=loss.to(torch.float32)
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                result = eval_model(model, validset_reader)
                logger.info('Dev acc: {0}'.format(result))
                if result >= best_pcc:
                    best_pcc = result
                    torch.save({'epoch': epoch,
                                'model': ori_model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_pcc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument('--model_name_or_path', required=True)
    # parser.add_argument('--bert_pretrain', required=True)
    # parser.add_argument('--post_pretrain', required=False)

    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument("--train_batch_size", default=1, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=4, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--max_len", default=12, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=3, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--seed",
                        type=int,
                        default=4321,
                        help="random seed for initialization")
    parser.add_argument("--device",
                        default='cuda:1',
                        help="cuda")

    parser.add_argument("--flag", default="train", type=str, help="choose to train or test")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    set_seed(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)

    if 'roformer' in args.model_name_or_path:
        # bert_model = BertForSequenceEncoder.from_pretrained(args.model_name_or_path)
        bert_model = RoFormerModel.from_pretrained(args.model_name_or_path, max_position_embeddings=args.max_len)
        tokenizer = RoFormerTokenizer.from_pretrained(args.model_name_or_path)
    else:
        bert_model = BertModel.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    logger.info('Start training!')
    logger.info("loading training set")
    train_dataset = data_loader(args.train_path, tokenizer, args.flag, args, batch_size=args.train_batch_size, )
    train_sampler = RandomSampler(train_dataset)
    trainset_reader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collect_fn
    )
    logger.info("loading validation set")
    valid_dataset = data_loader(args.valid_path, tokenizer, args.flag, args, batch_size=args.valid_batch_size,
                                test=True)
    sampler = SequentialSampler(valid_dataset)

    validset_reader = DataLoader(
        valid_dataset,
        sampler=sampler,
        batch_size=args.valid_batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=valid_dataset.collect_fn)
    logger.info('initializing estimator model')

    # bert_model = bert_model.cuda()
    bert_model = bert_model.to(args.device)

    ori_model = inference_model(bert_model, args)
    ori_model = ori_model.to(args.device)
    # model = nn.DataParallel(ori_model)
    # model = model.to(args.device)
    train_model(ori_model, ori_model, args, trainset_reader, validset_reader)

