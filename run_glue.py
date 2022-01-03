from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import json
import time
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from copy import deepcopy
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from bert_of_theseus import BertForSequenceClassification
from bert_of_theseus.replacement_scheduler import ConstantReplacementScheduler, LinearReplacementScheduler, MixedReplacementScheduler, ConstantThenLinearReplacementScheduler

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Prune():
    def __init__(
        self,
        model,
        pretrain_step=0,
        sparse_step=0,
        frequency=100,
        prune_dict={},
        restore_sparsity=False,
        fix_sparsity=False,
        balance='none', # when not use, set it to 'None'; else, 'fix'
        prune_device='default'):
        self._model = model
        self._t = 0
        self._initial_sparsity = {} # 0
        self._pretrain_step = pretrain_step # 1 epoch; but it is step
        self._sparse_step = sparse_step # 4 vs 16 epochs
        self._frequency = frequency # 20, 100 for sparse_step=4; when sparse_step=16, use the good frequency value (MRPC: 3.5k samples)
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._balance = balance
        self._prune_device = prune_device
        self._mask = {}

        self._prepare()

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                
                # print('name_para in _prepare:', name)

                if any(name == one for one in self._prune_dict):
                    if (self._balance == 'fix') and (len(parameter.shape) == 4) and (parameter.shape[1] < 4):
                        self._prune_dict.pop(name)
                        print("The parameter %s cannot be balanced pruned and will be deleted from the prune_dict." % name)
                        continue
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
                        self._initial_sparsity[name] = 1 - mask.sum().numpy().tolist() / weight.view(-1).shape[0]
                        self._mask[name] = mask
                    else:
                        self._initial_sparsity[name] = 0
                        self._mask[name] = torch.ones_like(weight)

    def _update_mask(self, name, weight, keep_k):
        if keep_k >= 1:
            thrs = torch.topk(weight.abs().view(-1), keep_k)[0][-1]
            mask = torch.where(weight.abs() >= thrs, torch.ones_like(weight), torch.zeros_like(weight))
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_fix_balance(self, name, weight, keep_k, inc_group):
        if keep_k >= 1:
            transpose_weight = weight.permute([0, 2, 1, 3])
            if transpose_weight.shape[-2] % inc_group == 0:
                mask = self._block_sparsity_balance(transpose_weight, keep_k, inc_group)
            else:
                temp1 = transpose_weight.shape[-2]
                temp4 = (inc_group - 1) * (temp1 // inc_group + 1)
                keep_k_1 = int(temp4 / temp1 * keep_k)
                keep_k_2 = keep_k - keep_k_1
                transpose_weight_1 = transpose_weight[:, :, :temp4, :]
                transpose_weight_2 = transpose_weight[:, :, temp4:, :]
                mask_1 = self._block_sparsity_balance(transpose_weight_1, keep_k_1, inc_group - 1)
                mask_2 = self._block_sparsity_balance(transpose_weight_2, keep_k_2, 1)
                mask = torch.cat([mask_1, mask_2], 1)
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _block_sparsity_balance(self, transpose_weight, keep_k, inc_group):
        reshape_weight = transpose_weight.reshape([-1, transpose_weight.shape[-2] * transpose_weight.shape[-1] // inc_group])
        base_k = keep_k // reshape_weight.shape[0]
        remain_k = keep_k % reshape_weight.shape[0]
        if remain_k > 0:
            thrs = torch.topk(reshape_weight.abs(), base_k + 1)[0][:, -1:]
        else:
            thrs = torch.topk(reshape_weight.abs(), base_k)[0][:, -1:]
        mask = torch.where(reshape_weight.abs() >= thrs, torch.ones_like(reshape_weight), torch.zeros_like(reshape_weight))
        mask = mask.view(transpose_weight.shape)
        mask = mask.permute([0, 2, 1, 3])
        return mask

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = self._pretrain_step < self._t < self._pretrain_step + self._sparse_step
        condition3 = (self._t - self._pretrain_step) % self._frequency == 0
        return condition1 and condition2 and condition3

    def _get_weight(self, parameter):
        if self._prune_device == 'default':
            weight = parameter.data
        elif self._prune_device == 'cpu':
            weight = parameter.data.to(device=torch.device('cpu'))
            # weight = parameter.to(device=torch.device('cpu'))
        return weight

    def prune(self):
        with torch.no_grad():
            self._t = self._t + 1
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._update_mask_conditions():
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (self._t - self._pretrain_step) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = target_sparsity + (self._initial_sparsity[name] - target_sparsity) * (1.0 - current_sparse_step / total_srarse_step) ** 3
                        keep_k = int(weight.view(-1).shape[0] * (1.0 - current_sparsity))
                        if self._balance == 'none':
                            self._update_mask(name, weight, keep_k)
                        elif self._balance == 'fix':
                            if len(weight.shape) != 4:
                                self._update_mask(name, weight, keep_k)
                            else:
                                self._update_mask_fix_balance(name, weight, keep_k, 4)
                    parameter.mul_(self._mask[name])

    # calculate sparsity rate
    def sparsity(self):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}
        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().numpy()
                total_param = total_param + temp.size
                total_nonezero = total_nonezero + np.flatnonzero(temp).size
                layer_sparse_rate[name] = 1 - np.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def train(args, train_dataset, model, teacher_model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # num_examples = 0
    if args.task_name == 'mrpc':
        num_examples = 3668
    elif args.task_name == 'cola':
        num_examples = 8551
    elif args.task_name == 'sts-b':
        num_examples = 5749
    elif args.task_name == 'rte':
        num_examples = 2490
    elif args.task_name == 'qnli':
        num_examples = 104743
    elif args.task_name == 'sst-2':
        num_examples = 67349
    print('task name: ', args.task_name)
    print(num_examples)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                    not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_pruning = len(train_dataloader) // args.gradient_accumulation_steps * 30
    scheduler_pruning = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_pruning)
    t_replacing_finetuning = len(train_dataloader) // args.gradient_accumulation_steps * 30
    scheduler_replacing_finetuning = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_replacing_finetuning)

    # Replace rate scheduler
    if args.scheduler_type == 'none':
        replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                replacing_rate=args.replacing_rate,
                                                                replacing_steps=args.steps_for_replacing)
    elif args.scheduler_type == 'linear':
        replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                              base_replacing_rate=args.replacing_rate,
                                                              k=args.scheduler_linear_k)
    elif args.scheduler_type == 'mixed':
        scheduler_linear_k = (1 - args.replacing_rate) / ( num_examples / 32 * 10 )
        replacing_rate_scheduler = MixedReplacementScheduler(bert_encoder=model.bert.encoder,
                                                             replacing_rate=args.replacing_rate,
                                                             k=scheduler_linear_k,
                                                             replacing_steps=args.steps_for_replacing)
    elif args.scheduler_type == 'constantLinearWithGap':
        replacing_rate_scheduler = ConstantThenLinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                             replacing_rate=args.replacing_rate,
                                                             base_replacing_rate=args.base_replacing_rate,
                                                             k=args.scheduler_linear_k,
                                                             replacing_steps=args.steps_for_replacing)
    

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        logger.warning("[BERT-of-Theseus] We haven't tested our model under multi-gpu. Please be aware!")
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        logger.warning("[BERT-of-Theseus] We haven't tested our model under distributed training. Please be aware!")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])


    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()
    train_examples = processor.get_train_examples(args.data_dir)


    num_steps_per_epoch =int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps)


    prune_dict = {}
    for k, v in model.named_parameters():
        if 'scc_layer' not in k:
            continue 
        # FF nn
        if ('intermediate.dense.weight' in k or 'output.dense.weight' in k) and ('attention.output.dense.weight' not in k):
            prune_dict[k] = args.sparsity
        # Att nn
        if 'attention.self.query.weight' in k or 'attention.self.key.weight' in k or 'attention.self.value.weight' in k or 'attention.output.dense.weight' in k:
            prune_dict[k] = args.sparsity

    freq = int((num_examples / 32) * args.num_prune_epochs / 12)   # prune 12 times
    # freq = 100
    prune = Prune(model, num_steps_per_epoch * 0, num_steps_per_epoch * args.num_prune_epochs, freq, prune_dict, False, False, 'none')

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    lr=[]
    lr_list = []

    # Prepare loss functions
    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    student_eval_results=[]
    for epoch_ in train_iterator:
        if epoch_ < args.num_prune_epochs:
            scheduler=scheduler_pruning
        else:
            scheduler=scheduler_replacing_finetuning
        print("learning rate: ", optimizer.param_groups[0]["lr"])
        # lr_list.append(optimizer.param_groups[0]["lr"])
        
        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.
        # nb_tr_examples = 0
        # nb_tr_steps = 0
        # device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            teacher_model.eval()
            # args.device="1"
            # args.device = "cuda:1"
            print("device: ", args.device)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            label_ids = inputs['labels']
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            
            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.

            with torch.no_grad():
                teacher_loss, teacher_logits, teacher_reps, teacher_atts = teacher_model(**inputs)

            student_loss, student_logits, student_reps, student_atts = model(**inputs)
            print("teacher_loss: ", teacher_loss)
            print("student_loss: ", student_loss)


            # loss = outputs[0]
            # loss = student_loss  # theseus loss

            # if not args.pred_distill:
            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)

            # print('len(teacher_atts), len(teacher_reps)', len(teacher_atts), len(teacher_reps))
            # print('')
            # print('len(student_atts), len(student_reps)', len(student_atts), len(student_reps))

            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)
            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

            loss_mse = MSELoss()
            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                            student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                            teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                att_loss += tmp_loss

            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
            new_student_reps = student_reps
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss

            loss = rep_loss + att_loss
            tr_att_loss += att_loss.item()
            tr_rep_loss += rep_loss.item()
            # else:
            if args.output_mode == "classification":
                cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                teacher_logits / args.temperature)
            elif args.output_mode == "regression":
                loss_mse = MSELoss()
                cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
            
            # no-aug add hard label loss
            loss = loss + cls_loss
            print("***************** print loss ********************")
            print("attention loss: ", att_loss)
            print("hidden state loss: ", rep_loss)
            print("soft logits loss: ", cls_loss)
            print("hard logits loss (loss compared to GT): ", student_loss)
            print("total loss: ", loss)

            
            tr_cls_loss += cls_loss.item()
            print("args.n_gpu: ", args.n_gpu)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                lr_list.append((epoch_, scheduler.last_epoch, scheduler.get_last_lr()[0], optimizer.param_groups[0]["lr"]))
                replacing_rate_scheduler.step()  # Update replace rate scheduler
                model.zero_grad()
                global_step += 1
                prune.prune()

                if step == 3439 or step == 4606:
                    logs = {}
                    if step == 3439:
                        print("************** sparse pruning finished ************** ")
                        print("step: ", step)
                    elif step == 4606:
                        print("************** replaceing finished ************** ")
                        print("step: ", step)
                    model.eval()
                    print("************** evaluate teacher model **************")
                    results = evaluate(args, teacher_model, tokenizer)
                    print("************** evaluate student model **************")
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        eval_key = 'eval_{}'.format(key)
                        logs[eval_key] = float(value)
                    model.train()

                

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        model.eval()
                        print("************** evaluate teacher model **************")
                        teacher_results = evaluate(args, teacher_model, tokenizer)
                        print("************** evaluate student model **************")
                        student_results = evaluate(args, model, tokenizer)
                        student_eval_results.append(step)
                        student_eval_results.append(student_results)

                        # for key, value in results.items():
                        #     eval_key = 'eval_{}'.format(key)
                        #     logs[eval_key] = float(value)
                        model.train()

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model,
                #                                             'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if (global_step + 1) % args.eval_step == 0:
                layer_sparse_rate, total_sparse_rate = prune.sparsity()
                print('')
                print('epoch %d: weight sparsity=%s; layer weight sparsity=%s' % (epoch_, total_sparse_rate, layer_sparse_rate))
                print('')

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    print("lr_list: ", lr_list)
    print("student_eval_results: ", student_eval_results)
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", 
                        type=str, 
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", 
                        type=str, 
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--task_name",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--replacing_rate", type=float, 
                        help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    parser.add_argument("--base_replacing_rate", type=float, 
                        help="Base replacing rate for linear")
    parser.add_argument("--scheduler_type", choices=['none', 'linear', 'mixed', 'constantLinearWithGap'], help="Scheduler function.")
    parser.add_argument("--scheduler_linear_k", default=0, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--steps_for_replacing", default=0, type=int,
                        help="Steps before entering successor fine_tuning (only useful for constant replacing)")

    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--sparsity", default=0.0, type=float, 
                        help="Pruning sparsity")
    parser.add_argument("--evaluate_during_training", default=True, action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1.6e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_prune_epochs", default=30.0, type=float,
                        help="Total number of pruning epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=True, action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    # added arguments
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--temperature',
                    type=float,
                    default=1.)
    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        # args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        # device = torch.device("cuda:1", args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config_class, teacher_model_class, tokenizer_class = TEACHER_MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_hidden_states = True
    config.output_attentions = True
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    # teacher model initialization
    teacher_model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    scc_n_layer = teacher_model.bert.encoder.scc_n_layer
    teacher_model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(teacher_model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
    # print("------------- Initial model --------------------")
    # para = {}
    # for k, v in teacher_model.named_parameters():
    #     para[k] = v

    
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    # Initialize successor BERT weights
    scc_n_layer = model.bert.encoder.scc_n_layer
    model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
    # print("length of model.bert.encoder.layer: ", len(model.bert.encoder.layer))
    # print("model.bert.encoder.layer: ", model.bert.encoder.layer)
    # print("length of scc_n_layer: ", len(scc_n_layer))
    # print("scc_n_layer: ", scc_n_layer)
    # for name, weight in model.named_parameters():
    #     if "bias" not in name:
    #         print("name: ", name)
    #         print(weight)

    # print("------------- model after initializing successor BERT weights --------------------")
    # for k, v in model.named_parameters():
    #     print(k)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    teacher_model.to(device)



    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, teacher_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)

    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)

    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    #     model.to(args.device)

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

    #         model = model_class.from_pretrained(checkpoint)
    #         # print("------------- final model --------------------")
    #         # for k, v in model.named_parameters():
    #         #     print(k, v)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=prefix)
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)


    # return results


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Total time: ", toc - tic)