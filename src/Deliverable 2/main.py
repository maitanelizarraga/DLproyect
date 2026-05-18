from load_data import get_data_loaders
from EDA import perform_eda 

def main():

    # 1. We get the loaders and the list of classes
        # batch_size is set to 64 (hyperparameter) for no collapse, stable learning and velocity
    loaders, classes = get_data_loaders(batch_size=64)
        # division of loaders
    train_loader, val_loader, test_loader = loaders

    print(f"\nInitializing Exploratory Data Analysis (EDA) ---")
    print(f"Classes to analyze: {classes}")
        #we make the analisys with the training dataset
    train_ds = train_loader.dataset 
    
    perform_eda(train_ds, train_loader)

if __name__ == "__main__":
    main()

