import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class IntentModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.

        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, target_size)
    
    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    def forward(self, inputs, targets):
        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """
        outputs = self.encoder(**inputs)
        hidden = outputs.last_hidden_state[:, 0, :]
        hidden = self.dropout(hidden)
        logit = self.classify(hidden)
        return logit

class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit


class CustomModel(IntentModel):
    def __init__(self, args, tokenizer, target_size, reinit_n_layers=0):
        super().__init__(args, tokenizer, target_size)
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size
        print(reinit_n_layers)

        # task1: add necessary class variables as you wish.
        if reinit_n_layers > 0:
            self._do_reinit(self.encoder, reinit_n_layers)

        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, target_size)

    
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) 

    def _do_reinit(self, model, reinit_n_layers):
        for i in range(reinit_n_layers):
            model.encoder.layer[-(i+1)].apply(self._init_weight_and_bias) 

class SupConModel(IntentModel):
    def __init__(self, args, tokenizer, target_size, feat_dim=768):
        super().__init__(args, tokenizer, target_size)
        # task1: initialize a linear head layer
        # this linear head is to be dropped after the contrastive learning is done. 
        self.head = nn.Linear(feat_dim, feat_dim)
        
        # this classifier is to be used for downstream tasks
        #self.classify = Classifier(args, target_size)
        self.classify = nn.Linear(feat_dim,target_size)
        # this dropout layer is to be used for contrastive learning for different augmentations
        self.dropout = nn.Dropout(args.drop_rate)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.xavier_uniform_(self.classify.weight)
        
    
    def freeze_contrastive(self):
        # task1: freeze the encoder and the linear head layer
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False
        for param in self.dropout.parameters():
            param.requires_grad = False
        for param in self.classify.parameters():
            param.requires_grad = True
        self.classify.train()

 
    def forward(self, inputs, targets, contrastive=False):

        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the normalized output of the dropout layer to the linear head layer; return the embedding
        """
        if contrastive:
            outputs = self.encoder(**inputs)
            hidden = outputs.last_hidden_state[:, 0, :]
            #first dropout for contrastive learning
            hidden1 = self.dropout(hidden)
            #second dropout for contrastive learning
            hidden2 = self.dropout(hidden)
            #normalize the hidden states
            feature1 = self.head(F.normalize(hidden1, p=2, dim=1))
            feature2 = self.head(F.normalize(hidden2, p=2, dim=1))
            #concatenate the two normalized hidden states so that it has shape [batchsize, 2, 768]
            features = torch.cat([feature1.unsqueeze(1), feature1.unsqueeze(1)], dim=1)
            #return features
        
            #might not be needed
            embedding = self.head(features)
            return embedding
        
        else:
            outputs = self.encoder(**inputs)
            hidden = outputs.last_hidden_state[:, 0, :]
            #hidden = self.dropout(hidden)
            logit = self.classify(hidden)
            return logit
