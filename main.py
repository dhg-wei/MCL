"""Training example.

Modified from https://github.com/kohjingyu/fromage/blob/main/main.py.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pickle as pkl
from fromage import data
from fromage import losses as losses_utils
from fromage import models
from fromage import utils
from fromage import evaluate
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

# Disable HuggingFace tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Available LLM models.
llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b',
              'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b',
              'facebook/opt-66b']
datasets = ['cc3m']
best_score = 0  # Variable to keep track of best model so far.


def parse_args(args):
  parser = argparse.ArgumentParser(description='FROMAGe training')
  parser.add_argument('--opt-version', default='facebook/opt-6.7b',
            choices=llm_models,
            help='OPT versions: ' +
              ' | '.join(llm_models) +
              ' (default: "facebook/opt-6.7b")')
  parser.add_argument('--visual-model', default='openai/clip-vit-large-patch14', type=str,
                      help="Visual encoder to use.")
  parser.add_argument('-d', '--dataset', metavar='DATASET',  help='Delimited list of datasets:' +
                      ' | '.join(datasets), default='cc3m',
                      type=lambda s: [x for x in s.split(',')])

  parser.add_argument('--val-dataset', metavar='DATASET', default='cc3m',
            type=lambda s: [x for x in s.split(',')],
            help='Validation dataset: ' +
              ' | '.join(datasets) +
              ' (default: cc3m)')
  parser.add_argument('--dataset_dir', default='datasets', type=str,
            help='Dataset directory containing .tsv files.')
  parser.add_argument('--image-dir', default='./data/', type=str,
            help='Dataset directory containing image folders.')
  parser.add_argument('--log-base-dir', default='./runs/', type=str,
            help='Base directory to write logs and ckpts to.')
  parser.add_argument('--exp_name', default='frozen', type=str,
            help='Name of experiment, used for saving checkpoints.')

  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
  parser.add_argument('--epochs', default=10, type=int, metavar='N',
            help='number of total epochs to run')
  parser.add_argument('--steps-per-epoch', default=1500, type=int, metavar='N',
            help='number of training steps per epoch')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
  parser.add_argument('--val-steps-per-epoch', default=-1, type=int, metavar='N',
            help='number of validation steps per epoch.')
  parser.add_argument('-b', '--batch-size', default=180, type=int,
            metavar='N',
            help='mini-batch size (default: 180), this is the total '
               'batch size of all GPUs on the current node when '
               'using Data Parallel or Distributed Data Parallel')
  parser.add_argument('--val-batch-size', default=None, type=int)
  parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
            metavar='LR', help='initial learning rate', dest='lr')
  parser.add_argument('--lr-warmup-steps', default=1000, type=int,
            metavar='N', help='Number of steps to warm up lr.')
  parser.add_argument('--lr-schedule-step-size', default=5, type=int,
            metavar='N', help='Number of steps before decaying lr.')
  parser.add_argument('--lr-schedule-gamma', default=0.1, type=float,
            metavar='N', help='Decay parameter for learning rate scheduler.')
  parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                      help='number of gradient accumulation steps')
  parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping amount')

  parser.add_argument('--precision', default='fp32', type=str, choices=['fp32', 'fp16', 'bf16'], help="Precision to train in.")
  parser.add_argument('--cap-loss-scale', type=float, default=1.0, help="Scale on captioning loss.")
  parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

  parser.add_argument('--concat-captions-prob', type=float, default=0.5, help="Probability of concatenating two examples sequentially for captioning.")
  parser.add_argument('--concat-for-ret', action='store_true', default=False, help="Whether to concatenate examples for retrieval mode.")
  parser.add_argument('--input-prompt', default=None, type=str, help="Input prompt for the language model, if any.")

  parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')
  parser.add_argument('--use_image_embed_norm', action='store_true', default=False, help="Whether to use norm on the image embeddings to make them equal to language.")
  parser.add_argument('--image_embed_dropout_prob', type=float, default=0.0, help="Dropout probability on the image embeddings.")
  parser.add_argument('--use_text_embed_layernorm', action='store_true', default=False, help="Whether to use layer norm on the text embeddings for retrieval.")
  parser.add_argument('--text_embed_dropout_prob', type=float, default=0.0, help="Dropout probability on the text embeddings.")
  parser.add_argument('--shared-emb-dim', default=256, type=int, metavar='N', help='Embedding dimension for retrieval.')
  parser.add_argument('--text-emb-layers', help='Layer to use for text embeddings. OPT-2.7b has 33 layers.', default='-1',
                      type=lambda s: [int(x) for x in s.split(',')])

  parser.add_argument('--max-len', default=24, type=int,
            metavar='N', help='Maximum length to truncate captions / generations to.')
  parser.add_argument('--n-visual-tokens', default=1, type=int,
            metavar='N', help='Number of visual tokens to use for the Frozen model.')

  parser.add_argument('--beta1', default=0.9, type=float, metavar='M', help='beta1 for Adam')
  parser.add_argument('--beta2', default=0.95, type=float, metavar='M', help='beta2 for Adam')
  parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
            metavar='W', help='weight decay (default: 0.0)', dest='weight_decay')
  parser.add_argument('-p', '--print-freq', default=10, type=int,
            metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
  parser.add_argument('--world-size', default=-1, type=int,
            help='number of nodes for distributed training')
  parser.add_argument('--rank', default=-1, type=int,
            help='node rank for distributed training')
  parser.add_argument("--local-rank", type=int, default=0)
    
  parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
            help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str,
            help='distributed backend')
  parser.add_argument('--seed', default=None, type=int,
            help='seed for initializing training. ')
  parser.add_argument('--gpu', default=None, type=int,
            help='GPU id to use.')
  parser.add_argument('--negbank', action='store_true',
            help='negtives training.')
  parser.add_argument('--qaprefix', action='store_true',
            help='qaprefix training.')

  parser.add_argument('--masktc', action='store_true',
            help='qaprefix training.')
  parser.add_argument('--Llama', action='store_true',
            help='llama2.')
  parser.add_argument('--allloss', action='store_true',
            help='llama2.')
  parser.add_argument('--b16', action='store_true',
            help='qaprefix training.')
  parser.add_argument('--LenRET', default=1, type=int, help='Size of images.')
  parser.add_argument('--multiprocessing-distributed', action='store_true',
            help='Use multi-processing distributed training to launch '
               'N processes per node, which has N GPUs. This is the '
               'fastest way to use PyTorch for either single node or '
               'multi node data parallel training')
  
  return parser.parse_args(args)


def main(args):
  args = parse_args(args)
  i = 1
  args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
  while os.path.exists(args.log_dir):
    args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
    i += 1
  os.makedirs(args.log_dir)

  with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
    json.dump(vars(args), wf, indent=4)

  with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
    utils.dump_git_status(out_file=wf)

  print(f'Logging to {args.log_dir}.')

  if args.seed is not None:
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
            'disable data parallelism.')

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)

def collate_fn(batch):
  new_dic = {}
  keys = batch[0].keys()
  for key in keys:
    if key =='neg_clip_tokens':
      new_dic[key] =None
      continue
      new_dic[key] =  torch.cat([x[key] for x in batch])
    elif key =='image_path':
      new_dic[key]=[x[key] for x in batch]
    else:
      new_dic[key] =  torch.stack([x[key] for x in batch])

  return new_dic

def all_gather_nd(tensor):
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[:size[0]])
    return all_tensors
  
def main_worker(gpu, ngpus_per_node, args):
  """Setup code."""
  global best_score
  args.gpu = gpu

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                world_size=args.world_size, rank=args.rank)

  # Create model
  model_args = models.FrozenArgs()
  model_args.opt_version = args.opt_version
  model_args.freeze_lm = True
  model_args.visual_encoder = args.visual_model
  model_args.freeze_vm = True
  model_args.n_visual_tokens = args.n_visual_tokens
  model_args.use_image_embed_norm = args.use_image_embed_norm
  model_args.image_embed_dropout_prob = args.image_embed_dropout_prob
  model_args.use_text_embed_layernorm = args.use_text_embed_layernorm
  model_args.text_embed_dropout_prob = args.text_embed_dropout_prob
  model_args.shared_emb_dim = args.shared_emb_dim
  model_args.text_emb_layers = args.text_emb_layers
  model_args.LenRET = args.LenRET
  model_args.masktc = args.masktc

  model_args.Llama = args.Llama
  if args.Llama:
    model_args.Llama = True
    tokenizer = LlamaTokenizer.from_pretrained("fromage/llama/hugging-llama-2-7b")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  else:
    model_args.Llama = False
    tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False,local_files_only=False)
  # Add an image token for loss masking (and visualization) purposes.
  tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer
  print('Adding [RET] token to vocabulary.')
  
  if args.LenRET==1:
    print('Before adding new token, tokenizer("[RET]") =', tokenizer('[RET]', add_special_tokens=False))
    num_added_tokens = tokenizer.add_tokens('[RET]')
    print(f'After adding {num_added_tokens} new tokens, tokenizer("[RET]") =', tokenizer('[RET]', add_special_tokens=False))
    ret_token_idx = tokenizer('[RET]', add_special_tokens=False).input_ids
    assert len(ret_token_idx) == 1, ret_token_idx
    model_args.retrieval_token_idx = ret_token_idx[0]
    args.retrieval_token_idx = ret_token_idx[0]
  else:
    model_args.retrieval_token_idx = []
    args.retrieval_token_idx = []
    for i in range(args.LenRET):
      RET_Token = f'[RET{i}]'
      print(f'Before adding new token, tokenizer("[RET{i}]") =', tokenizer(RET_Token, add_special_tokens=False))
      num_added_tokens = tokenizer.add_tokens(RET_Token)
      print(f'After adding {num_added_tokens} new tokens, tokenizer("[RET{i}]") =', tokenizer(RET_Token, add_special_tokens=False))
      ret_token_idx = tokenizer(f'[RET{i}]', add_special_tokens=False).input_ids
      assert len(ret_token_idx) == 1, ret_token_idx
      model_args.retrieval_token_idx.append(ret_token_idx[0])
      args.retrieval_token_idx.append(ret_token_idx[0])

  # Save model args to disk.
  with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
    json.dump(vars(model_args), f, indent=4)

  model = models.Fromage(tokenizer, model_args)
  if args.precision == 'fp16':
    model = model.half()
  elif args.precision == 'bf16':
    model = model.bfloat16()

  # Print parameters and count of model.
  param_counts_text = utils.get_params_count_str(model)
  with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
    f.write(param_counts_text)

  # Log trainable parameters to Tensorboard.
  _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)

  if not torch.cuda.is_available():
    print('WARNING: using CPU, this will be slow!')
    model = torch.nn.DataParallel(model)
  elif args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      model.cuda(args.gpu)
      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs of the current node.

      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
      model.cuda()
      # DistributedDataParallel will divide and allocate batch_size to all
      # available GPUs if device_ids are not set
      model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
  elif args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
  else:
    model = torch.nn.DataParallel(model).cuda()

  # define loss function (criterion), optimizer, and learning rate scheduler
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  optimizer_cls = torch.optim.AdamW
  print('Using torch.optim.AdamW as the optimizer.')
  optimizer = optimizer_cls(model.parameters(), args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
                eps=1e-8)

  """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
  scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch, gamma=args.lr_schedule_gamma)
  scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)
  
  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      if args.gpu is None:
        checkpoint = torch.load(args.resume)
      else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
      args.start_epoch = checkpoint['epoch']
      best_score = checkpoint['best_score']

      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  train_dataset = data.get_dataset(args, 'train', tokenizer)

  print(f'Training with {len(train_dataset)} examples.')

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)

  else:
    train_sampler = None
    

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler,collate_fn=collate_fn)


  with open('./features/text_embs_CLIP_L14.pkl' ,'rb') as f:
    cc3m_text_embs = pkl.load(f)

  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      train_sampler.set_epoch(epoch)

    train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args, cc3m_text_embs)
    is_best = False

    if True:
      if not args.multiprocessing_distributed or (args.multiprocessing_distributed
          and args.rank % ngpus_per_node == 0):
        utils.save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_score': 0,
          'optimizer' : optimizer.state_dict(),
          'scheduler' : scheduler.state_dict()
        }, is_best, os.path.join(args.log_dir, f'ckpt_epoch_{epoch + 1}'))


def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args,cc3m_text_embs):
  """Main training loop."""
  ngpus_per_node = torch.cuda.device_count()
  batch_time = utils.AverageMeter('Time', ':6.3f')

  data_time = utils.AverageMeter('Data', ':6.3f')
  losses = utils.AverageMeter('Loss', ':.4e')
  ce_losses = utils.AverageMeter('CeLoss', ':.4e')
  top1 = utils.AverageMeter('Acc@1', ':6.2f')
  top5 = utils.AverageMeter('Acc@5', ':6.2f')
  cont_losses = utils.AverageMeter('ContLoss', ':.4e')
  top1_caption = utils.AverageMeter('AccCaption@1', ':6.2f')
  top5_caption = utils.AverageMeter('AccCaption@5', ':6.2f')
  top1_image = utils.AverageMeter('AccImage@1', ':6.2f')
  top5_image = utils.AverageMeter('AccImage@5', ':6.2f')



  progress = utils.ProgressMeter(
    args.steps_per_epoch,
    [batch_time, losses, ce_losses, cont_losses, top1, top5,top1_image,top1_caption],
    prefix="Epoch: [{}]".format(epoch))

  # switch to train mode
  model.train()

  end = time.time()
  MSE_Loss = nn.MSELoss()
  for i, batch in enumerate(train_loader):
    (image_paths, images, caption_images, tgt_tokens, token_len, tc_tokens, tc_len,new_caption_clip_tokens,cc_tokens,cc_len,pc_len,masks_tc,text_condition_clip_tokens) = batch.values()
#     torch.autograd.set_detect_anomaly(True)
    actual_step = epoch * args.steps_per_epoch + i + 1
    # measure data loading time
    data_time.update(time.time() - end)

    if torch.cuda.is_available():
      images = images.cuda(args.gpu, non_blocking=True)
      tgt_tokens = tgt_tokens.cuda(args.gpu, non_blocking=True)
      token_len = token_len.cuda(args.gpu, non_blocking=True)
      masks_tc= masks_tc.cuda(args.gpu, non_blocking=True)
      text_condition_clip_tokens=text_condition_clip_tokens.cuda(args.gpu, non_blocking=True)
    if args.precision == 'fp16':
      images = images.half()
    elif args.precision == 'bf16':
      images = images.bfloat16()

    model_modes=[]

    model_modes = ['mc_ret','mc_cap','cap','ret']
    

    if torch.cuda.is_available():
      tc_tokens = tc_tokens.cuda(args.gpu, non_blocking=True)
      tc_len = tc_len.cuda(args.gpu, non_blocking=True)
      cc_tokens = cc_tokens.cuda(args.gpu, non_blocking=True)
      cc_len = cc_len.cuda(args.gpu, non_blocking=True)
      pc_len = pc_len.cuda(args.gpu, non_blocking=True)
      new_caption_clip_tokens = new_caption_clip_tokens.cuda(args.gpu, non_blocking=True)

    for model_mode in model_modes:
      loss = 0
      mode_start = time.time()

      concat_captions = np.random.uniform(0, 1) < args.concat_captions_prob
      if not args.concat_for_ret:
        concat_captions = concat_captions and model_mode == 'cap'
      if model_mode == 'mc_ret':
        (model_output, full_labels, last_embedding, visual_embs) = model(
          images, tc_tokens, tc_len, mode=model_mode, concat_captions=concat_captions, inference=False,
          tc_tokens = tc_tokens,tc_len=tc_len, new_caption_clip_tokens=new_caption_clip_tokens,masks_tc=masks_tc,text_condition_clip_tokens=text_condition_clip_tokens)
      elif model_mode == 'mc_cap':
        (model_output, full_labels, last_embedding, visual_embs) = model(
          images, cc_tokens, cc_len,pc_len=pc_len, mode=model_mode, concat_captions=concat_captions, inference=False)
      else:
        (model_output, full_labels, last_embedding, visual_embs) = model(
          images, tgt_tokens, token_len, mode=model_mode, concat_captions=concat_captions, inference=False,masks_tc=masks_tc)
      output = model_output.logits

      if model_mode == 'mc_ret':

        # Cross replica concat for embeddings.
        if args.distributed:
          # print(visual_embs.shape)
          all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
          all_last_embedding = [torch.zeros_like(last_embedding) for _ in range(dist.get_world_size())]
          dist.all_gather(all_visual_embs, visual_embs)
          dist.all_gather(all_last_embedding, last_embedding)
          # Overwrite with embeddings produced on this replace, which have the gradient.
          # print(dist.get_world_size(),dist.get_rank())
          all_visual_embs[dist.get_rank()] = visual_embs
          all_last_embedding[dist.get_rank()] = last_embedding
          visual_embs = torch.cat(all_visual_embs)
          last_embedding = torch.cat(all_last_embedding)



          start_idx = args.rank * images.shape[0]
          end_idx = start_idx + images.shape[0]
  
          logits_per_image = visual_embs @ last_embedding.t()

          logits_per_text = logits_per_image.t()
          if i == 0:
            print(f'Running contrastive loss over logits_per_text.shape = {logits_per_text.shape} and logits_per_image.shape = {logits_per_image.shape}')

          if args.negbank:

            random_idx = torch.randperm(cc3m_text_embs.size(0))[:50000]
            random_neg = cc3m_text_embs[random_idx]       
            random_neg/=random_neg.norm(dim=-1,keepdim=True)
            random_neg = random_neg*(torch.ones([]) * 14.2)
            random_neg = random_neg.to(logits_per_image.device).to(last_embedding.dtype)

            l_pos = torch.einsum("nc,nc->n", [last_embedding, visual_embs]).unsqueeze(-1)
            l_neg = torch.einsum("nc,ck->nk", [last_embedding, random_neg.T.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)


            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
            image_loss = losses_utils.contrastive_loss_neg(logits, labels)
            caption_loss = losses_utils.contrastive_loss(logits_per_image)
            image_acc1, image_acc5 = losses_utils.contrastive_acc_neg(logits, topk=(1, 5))
            caption_acc1, caption_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))

          else:
          # Compute contrastive losses for retrieval.
            caption_loss = losses_utils.contrastive_loss(logits_per_text)
            image_loss = losses_utils.contrastive_loss(logits_per_image)
            caption_acc1, caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
            image_acc1, image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))


          loss += args.ret_loss_scale * (caption_loss + image_loss) / 2.0 

          cont_losses.update(loss.item(), images.size(0))

          # measure accuracy and record loss
          top1_caption.update(caption_acc1[0], images.size(0))
          top5_caption.update(caption_acc5[0], images.size(0))
          top1_image.update(image_acc1[0], images.size(0))
          top5_image.update(image_acc5[0], images.size(0))  
      
      
      if model_mode == 'cap':
        acc1, acc5 = utils.accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
      if model_mode == 'mc_cap':
        acc1, acc5 = utils.accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
      ce_loss = model_output.loss
      if model_mode == 'cap':
        ce_loss = ce_loss * args.cap_loss_scale
      elif model_mode == 'mc_cap':
        ce_loss = ce_loss * args.cap_loss_scale
      elif model_mode == 'ret':
        ce_loss = ce_loss * args.ret_loss_scale * 0
      elif model_mode == 'mc_ret':
        ce_loss = ce_loss * 0
      else:
        raise NotImplementedError

      loss += ce_loss
      if not model_mode=='mc_ret':
        ce_losses.update(ce_loss.item(), images.size(0))

      if model_mode == 'ret':
        # Cross replica concat for embeddings.
        if len(last_embedding.shape)==3:
          import random
          x_l = random.randint(0,4)
          last_embedding = last_embedding[:,x_l,:].contiguous()
        
        if args.distributed:
          all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
          all_last_embedding = [torch.zeros_like(last_embedding) for _ in range(dist.get_world_size())]
          dist.all_gather(all_visual_embs, visual_embs)
          dist.all_gather(all_last_embedding, last_embedding)
          # Overwrite with embeddings produced on this replace, which have the gradient.
          all_visual_embs[dist.get_rank()] = visual_embs
          all_last_embedding[dist.get_rank()] = last_embedding
          visual_embs = torch.cat(all_visual_embs)
          last_embedding = torch.cat(all_last_embedding)

          start_idx = args.rank * images.shape[0]
          end_idx = start_idx + images.shape[0]
  
          
          logits_per_image = visual_embs @ last_embedding.t()
          logits_per_text = logits_per_image.t()
          if i == 0:
            print(f'Running contrastive loss over logits_per_text.shape = {logits_per_text.shape} and logits_per_image.shape = {logits_per_image.shape}')

          # Compute contrastive losses for retrieval.
          caption_loss = losses_utils.contrastive_loss(logits_per_text)
          image_loss = losses_utils.contrastive_loss(logits_per_image)
          caption_acc1, caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
          image_acc1, image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
          loss += args.ret_loss_scale * (caption_loss + image_loss) / 2.0
          cont_losses.update(loss.item(), images.size(0))





      loss = loss / args.grad_accumulation_steps
      losses.update(loss.item(), images.size(0))
      loss.backward()


      if ((i + 1) % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
        # Zero out gradients of the embedding matrix outside of [RET].
        for param in model.module.model.input_embeddings.parameters():
          assert param.grad.shape[0] == len(tokenizer)
          # Keep other embeddings frozen.
  #         mask = torch.arange(param.grad.shape[0]) != args.retrieval_token_idx
          mask = torch.isin(torch.arange(param.grad.shape[0]), torch.tensor(args.retrieval_token_idx))
          param.grad[~mask, :] = 0
  
        # compute gradient and do SGD step
        if args.grad_clip > 0:
          nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if actual_step == 1 or (i + 1) % args.print_freq == 0:
      ex_per_sec = args.batch_size / batch_time.avg
      if args.distributed:
        batch_time.all_reduce()
        data_time.all_reduce()
        ex_per_sec = (args.batch_size / batch_time.avg) * ngpus_per_node

        losses.all_reduce()
        
        # ce_losses.all_reduce()
        # top1.all_reduce()
        # top5.all_reduce()
        # ret_time.all_reduce()
        # cont_losses.all_reduce()
        # top1_caption.all_reduce()
        # top5_caption.all_reduce()
        # top1_image.all_reduce()
        # top5_image.all_reduce()
        # cap_time.all_reduce()

      progress.display(i + 1)


      batch_time.reset()

      data_time.reset()
      losses.reset()
      ce_losses.reset()
      top1.reset()
      top5.reset()
      cont_losses.reset()
      top1_caption.reset()
      top5_caption.reset()
      top1_image.reset()
      top5_image.reset()

    if i == args.steps_per_epoch - 1:
      break

    scheduler.step()
    curr_lr = scheduler.get_last_lr()



if __name__ == '__main__':
  main(sys.argv[1:])