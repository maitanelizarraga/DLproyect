import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend 
from torchmetrics.classification import MulticlassF1Score
import os

from load_data import get_data_loaders, device

# Dinamic Architecture (elastic neuronal network)
class DynamicCNN(nn.Module):
    def __init__(self, trial):
        super(DynamicCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        n_layers = trial.suggest_int("n_conv_layers", 2, 4) #optuna decides 2,3,4 layers for the network
        in_channels = 3 #red, green and blue colors of the images
        
        for i in range(n_layers):
            out_channels = trial.suggest_categorical(f"n_filters_l{i}", [16, 32, 64]) #number of filters in specific layer
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) #search patterns in grid 3x3 and border 1 pix
            self.layers.append(nn.BatchNorm2d(out_channels)) #normalize
            self.layers.append(nn.ReLU()) 
            self.layers.append(nn.MaxPool2d(2, 2)) #reduction size for foccus
            in_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) #reduce 
        self.fc = nn.Linear(in_channels, 2) #select if pneum or no
        
        init_type = trial.suggest_categorical("weight_init", ["kaiming", "xavier"]) #selects the best(fastest learning) model 
        self.apply(lambda m: self._init_weights(m, init_type))

    #best weight depending on model
    def _init_weights(self, m, init_type):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight)
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

# Optuna Objective Function
def objective(trial):
    #configuration of batch size(power of 2)
    n = trial.suggest_int("bsize_exp", 4, 6) 
    batch_size = 2 ** n
    trial.set_user_attr("bsize", batch_size) 
    

    loaders, _ = get_data_loaders(batch_size=batch_size)
    train_loader, val_loader, _ = loaders

    #dynamic neuronal network 
    model = DynamicCNN(trial).to(device)
    
    #optimizator configuration
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"]) #select best opt
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) #select learning rate
    
    if optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.85, 0.99) #select how much the past gradients incluence
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    else:
        momentum = trial.suggest_float("momentum", 0.5, 0.99) #for not get local valley
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    criterion = nn.CrossEntropyLoss() #error measure
    metric_f1 = MulticlassF1Score(num_classes=2, average='macro').to(device)
    

    #training and feedback
    epochs = 5 
    
    print(f"\\n---> Initializing Trial {trial.number} | Batch Size: {batch_size} | Opt: {optimizer_name} | LR: {lr:.5f}")
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() #resets past errors
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() #calcul better directions
            optimizer.step()
           
        #validation and pruning logic 
        model.eval()
        metric_f1.reset()
        with torch.no_grad(): #dont learn in exam
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                metric_f1.update(outputs, labels)
        
        val_f1 = metric_f1.compute().item()
        

        print(f"    Trial {trial.number} - Epoch [{epoch+1}/{epochs}] - Val F1: {val_f1*100:.2f}%")
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            print(f"    [!] Trial {trial.number} pruned because of poor performance.")
            raise optuna.exceptions.TrialPruned()

    return val_f1

# Study Configuration
if __name__ == "__main__":
    print("Initializing optimization with Optuna...")
    

    storage = JournalStorage(JournalFileBackend("optuna_journal.log"))
    
    #select hyperparameters with TPE(first 5 are randm) and prune with median(2epch)
    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = MedianPruner(n_warmup_steps=2, n_startup_trials=5)
    
 # In stead of JournalStorage, use a simple database
    storage_name = "sqlite:///optuna_study.db"

    study = optuna.create_study(
        study_name="cnn_optimization",
        storage=storage_name, 
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=15)
    
    print("\\n--- Optimization Completed! ---")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")



#        \n--- Optimization Completed! ---

# Best hyperparameters found:

#    bsize_exp: 4 (best batch size 2^4=16)

#    n_conv_layers: 4 (num of necessary layers)

#(num of filters in each layer)
#    n_filters_l0: 32 

#    n_filters_l1: 32

#    n_filters_l2: 16

#   n_filters_l3: 64

#    weight_init: xavier

#    optimizer: Adam

#    lr: 0.00031417639113887194 (learning rate slow but precise)

#    beta1: 0.9435547457342313 (high inertia to not distract)