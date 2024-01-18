import torch 
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.data import DataLoader, Dataset
import pickle 
import tqdm
from util.optim import Optim
def create_output_folder(folder_name:str):
    folder = os.path.join('output'+ time.localtime())
    print(folder)
    return folder

class Log_csv():
    def __init__(self, file):
        self.file=file
        
    def write_to_file(self, data):
        print(data)
        with open(self.file) as f:
            f.write(data)
            
    def write_from_file(self):
        with open(self.file, 'r') as f:
            for line in f.readlines():
                print(line)
        
    
class Training(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        self.args = args
        self.logging = create_output_folder('output')
        self.log_path = Log_csv.write_to_file(self.logging+'record.csv')
        self.log_csv = Log_csv.write_to_file(self.logging+'log.csv')
        
        if self.args.restore:  
            checkpoints = torch.load(self.args.restore)
            
        self.train_loader = DataLoader()
        
        with open(self.args.embedding, 'r') as f:
            embedding = pickle.load(f)
            embedding = torch.load(embedding)
            
        self.model = GPG(config=self.args, embedding=embedding).cuda()
        self.optim = Optim(self.args.optim, self.args.learning_rate, lr_decay=self.args.learning_rate_delay, momentum=self.args.momentum, start_decay_at=self.args.start_decay_at, decay_factor=self.args.decay_factor, patience=self.args.patience)
        
        if self.args.restore:
            self.optim = checkpoints['optim']
            self.model.load_state_dict(checkpoints['model'])
        else:
            self.optim = Optim.set_parameters(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optim)
        self.model = nn.DataParallel(self.model, device_ids=[0],dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        for epoch in range(self.args.train_epochs):
            train_dataloader = self.dataloader.train_loader
            self.model.train()
            running_loss=0
            
            if self.args.schedule:
                self.scheduler.step()
                print(f"Decaying learning rate to {self.scheduler.get_lr()[0]}")
                
            for i,batch in enumerate(tqdm(train_dataloader)):
                self.optim.zero_grad()
                logits, softmasks = self.model(batch)
                
                eos_trg = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']) 
                eos_trg = eos_trg[:, 1:].cuda()
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                
    def test_loss(self):
        
        test_dataloader = self.dataloader.dev_loader
        test_loss=0.0
        for epoch in range(self.args.train_epochs):
            i=0
            for i,batch in enumerate(tqdm(test_dataloader)):
                with torch.no_grad()
                    eos_trg = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']) 
                    eos_trg = eos_trg[:, 1:].cuda()
                    logits, softmasks = self.model(batch)
                    batch_size, nsteps, _ = logits.size()
                    preds = logits.view(batch_size * nsteps, -1)
                    targets = eos_trg.contiguous().view(-1).cuda()
                    loss = self.criterion(preds, targets)
                    test_loss += loss.data.item()
                    
            return test_loss  
        
    def save_model(self, path):
        model_state_dict = self.model.state_dict()
        if os.path.exists(path):
            self.path=path
        else:
            return FileNotFoundError(f"The specified path '{path}' does not exist.")
        
        checkpoints={
            'model':model_state_dict,
            'config':self.args,
            'optim':self.optim
        }
        torch.save(checkpoints,self.path)
                
            
    
            
        
        
        