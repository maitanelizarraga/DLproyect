from load_data import get_data_loaders
from EDA import perform_eda

def main():
    loaders, datasets = get_data_loaders(batch_size=64)
    
    train_loader, val_loader, test_loader = loaders
    train_ds, val_ds, test_ds = datasets

    perform_eda(train_ds, train_loader)

if __name__ == "__main__":
    main()