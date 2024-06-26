import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result

from adamw_bf16 import LR, AdamW_BF16
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import warnings
warnings.filterwarnings('ignore')

def train(model, data_loader, optimizer, epoch, scaler, device, mixed_precision = False):
    model.train()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f"Train Epoch: [{epoch}]"
    print_freq = 50

    if (mixed_precision):
        for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                loss = model(image, question, answer, train=True, n=n, weights = weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    else:
        for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image, weights = image.to(torch.bfloat16).to(device, non_blocking=True), weights.to(torch.bfloat16).to(device, non_blocking=True)
            loss = model(image, question, answer, train=True, n=n, weights = weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    print('Average stats:', metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 

@torch.no_grad()
def evaluation(model, data_loader, device, training_config):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    if training_config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(torch.bfloat16).to(device,non_blocking=True)             

        if training_config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            
            for answer, ques_id in zip(answers, question_id):
                ques_id = ques_id      
                result.append({"ground_truth":ques_id, "model_answer":answers})
            
        elif training_config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=training_config['k_test'])      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"ground_truth":ques_id, "model_answer":answer_list[answer_id]})   

    return result

def main(args, data_config, training_config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    if (device == torch.device('cuda')):
        print(f"Running on {torch.cuda.get_device_name(0)}")
    else:
        print(f"Running on cpu")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    ### Dataset ###
    print('Loading dataset')
    dataset = create_dataset(data_config, 'vqa_rad')

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(dataset, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(dataset,samplers,
                                              batch_size=[training_config['batch_size_train'],training_config['batch_size_test']],
                                              num_workers=[1,1],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    
    ### Model ###
    print('Loading model')
    model = None
    if (args.target == 'train'):
        model = blip_vqa(pretrained=training_config['pretrained'], image_size=training_config['image_size'], 
                       vit=training_config['vit'], vit_grad_ckpt=training_config['vit_grad_ckpt'], vit_ckpt_layer=training_config['vit_ckpt_layer'])
    else:
        model = blip_vqa(pretrained=training_config['checkpoint'], image_size=training_config['image_size'], 
                       vit=training_config['vit'], vit_grad_ckpt=training_config['vit_grad_ckpt'], vit_ckpt_layer=training_config['vit_ckpt_layer'])
    model = model.to(torch.bfloat16).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = AdamW_BF16(model.parameters(), lr_function=LR(lr=training_config['init_lr'], decay_power=training_config['power_decay']))
    scaler = torch.cuda.amp.GradScaler()

    best = 0
    best_epoch = 0 
    start_time = time.time()

    if args.target == 'train':

        print("Start training")    
        for epoch in range(0, training_config['max_epoch']):
            if not args.evaluate:        
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                    
                cosine_lr_schedule(optimizer, epoch, training_config['max_epoch'], training_config['init_lr'], training_config['min_lr'])
                    
                train_stats = train(model, train_loader, optimizer, epoch, scaler, device) 

            else:         
                break        
            
            if utils.is_main_process():     
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                        
                        
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler' : scaler.state_dict(),
                    'config': training_config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

            # dist.barrier()
    else:
        print("Start testing")
        if training_config['checkpoint'] == 'None':
            raise ValueError('checkpoint must be setted in testing mode')
        if args.target != 'test':
            raise ValueError('What you want to do ? --target ["train", "test"]')
        vqa_result = evaluation(model_without_ddp, test_loader, device, training_config)        
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Running time {}'.format(total_time_str)) 
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='train')
    parser.add_argument('--dataset_config', default='./configs/dataset.yaml') 
    parser.add_argument('--training_config', default='./configs/training.yaml') 
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    dataset_config = yaml.load(open(args.dataset_config, 'r'), Loader=yaml.Loader)
    training_config = yaml.load(open(args.training_config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(training_config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    print(f'Running in {args.target} mode!')
    main(args, dataset_config, training_config)
