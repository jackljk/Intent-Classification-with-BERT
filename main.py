import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories, plotGraph
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn
from transformers import get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer, plot=False):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split = 'train')

    # task2: setup model's optimizer_scheduler if you have
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # list to store accuracy and loss
    train_accs, val_accs = [], []
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        acc, losses = 0, 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()

            optimizer.step()  # backprop to update the weights
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
    
        val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        train_acc = acc/len(datasets['train'])
        print('epoch', epoch_count, '| losses:', losses, '| train acc:', acc/len(datasets['train']))

        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    if plot:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plotGraph(train_accs, val_accs,  s_dir='plots')
  
def custom_train(args, model, datasets, tokenizer, plot=False):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split = 'train')

    # task2: setup model's optimizer_scheduler if you have
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * args.n_epochs)
    
    # list to store accuracy and loss
    train_accs, val_accs = [], []

    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses, acc = 0, 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()

            optimizer.step()  # backprop to update the weights
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
    
        val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        train_acc = acc/len(datasets['train'])
        print('epoch', epoch_count, '| losses:', losses, '| train acc:', acc/len(datasets['train']))

        train_accs.append(train_acc)
        val_accs.append(val_acc)

    if plot:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plotGraph(train_accs, val_accs,  s_dir='plots')

def run_eval(args, model, datasets, tokenizer, split='validation'):
    criterion = nn.CrossEntropyLoss() 
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    losses = 0

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        loss = criterion(logits, labels)
        loss.backward()
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
        losses += loss.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'{split} loss:', losses, f'|dataset split {split} size:', len(datasets[split]))

    return acc/len(datasets[split])

def supcon_train(args, model, datasets, tokenizer, plot=False):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    
    # task2: setup optimizer_scheduler in your model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.contrast_learning_rate)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * 10)

    # task3: write a training loop for SupConLoss function 
    earlystop_loss = np.inf
    early_count = 0
    for epoch_count in range(args.contrast_n_epochs):
        losses = 0
        model.train()
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            features = model(inputs, labels, contrastive=True)
            if args.slimCLR:
                loss = criterion(features, labels)
            else:
                loss = criterion(features)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            losses += loss.item()
            model.zero_grad()
            
        if earlystop_loss < losses:
            early_count += 1
            if early_count > 3:
                print("early stopped: ",'contrastive','epoch', epoch_count, '| losses:', losses)
                break
        else:
            early_count = 0
        earlystop_loss = losses
        #print statements
        print('contrastive','epoch', epoch_count, '| losses:', losses)

    # freeze contrastive layers and train the classifier
    model.freeze_contrastive()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * args.n_epochs)

    # List for storing accuracy and loss
    train_accs, val_accs = [], []

    for epoch_count in range(args.n_epochs):
        losses, acc = 0, 0
        model.train()
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            
            optimizer.step()
            #scheduler.step()
            model.zero_grad()
            losses += loss.item()
        #print statements
        print('normal','epoch', epoch_count, '| losses:', losses, '| train acc:', acc/len(datasets['train']))
        val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        train_acc = acc/len(datasets['train'])

        train_accs.append(train_acc)
        val_accs.append(val_acc)

    if plot:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plotGraph(train_accs, val_accs,  s_dir='plots')


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))

    if args.task == 'baseline':
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'custom': # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60, reinit_n_layers=args.reinit_n_layers).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        custom_train(args, model, datasets, tokenizer, plot=True)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'supcon':
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer, plot=True)
        run_eval(args, model, datasets, tokenizer, split='test')
