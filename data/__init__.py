import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.vqa_dataset import vqa_dataset

def create_dataset(config, dataset_name):
    """
        config: list of link of dataset
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
    ])
    data = load_dataset(config[dataset_name])
    train_dataset = vqa_dataset(transform_train, data['train']['image'], data['train']['question'], data['train']['answer'], split='train')
    test_dataset = vqa_dataset(transform_test, data['test']['image'], data['test']['question'], data['test']['answer'], split='test')

    return train_dataset, test_dataset

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    