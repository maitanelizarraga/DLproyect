import torch
import torch.nn as nn
import kagglehub
import os
from torch.utils.data import DataLoader
from dataset import AudioNoiseDataset
from model import AudioTransformer
from main import get_splits 

def save_test_comparison(noisy, clean, predicted, save_dir="src/Deliverable3/images"):
    """Saves a final test result image to prove the model works on unseen data."""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    noisy_img = noisy[0, 0].cpu().detach().numpy()
    clean_img = clean[0, 0].cpu().detach().numpy()
    pred_img = predicted[0, 0].cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy_img, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Test Input (Noisy)")
    axes[1].imshow(clean_img, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Test Target (Clean)")
    axes[2].imshow(pred_img, aspect='auto', origin='lower', cmap='magma')
    axes[2].set_title("Model Final Prediction")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_test_comparison.png"))
    plt.close()
    print(f"Comparison image saved to {save_dir}/final_test_comparison.png")

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 2. Load Data Paths 
    print("Locating Test Data...")
    clean_dir = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    noise_dir = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
    
    _, _, clean_test = get_splits(clean_dir)
    _, _, noise_test = get_splits(noise_dir)

    # Use a solid slice of the test set 
    test_dataset = AudioNoiseDataset(clean_test[:1000], noise_test[:1000])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. Load the Optimized Model
    model = AudioTransformer(num_mels=64, d_model=128, num_layers=3).to(device)
    
    model_path = "src/Deliverable3/models_optimized/best_transformer_epoch_50.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")

    # 4. Final Evaluation Loop
    criterion = nn.MSELoss()
    total_test_loss = 0.0
    
    print("Running final inference on Test Set...")
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(test_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # Prediction
            predicted = model(noisy)
            
            # Calculate Loss
            loss = criterion(predicted, clean)
            total_test_loss += loss.item()
            
            # Save the very first batch as a visual example for the report
            if i == 0:
                save_test_comparison(noisy, clean, predicted)

    avg_test_loss = total_test_loss / len(test_loader)
    
    print("\n" + "="*30)
    print("      FINAL TEST RESULTS      ")
    print("="*30)
    print(f"Test MSE Loss: {avg_test_loss:.4f}") #Unseen Data' performance.
    print("="*30)

if __name__ == "__main__":
    main()