import torch
import numpy as np
import matplotlib.pyplot as plt
import timm
import os
from load_data import get_data_loaders
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Function to un-normalize the image so we can plot it
def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating Grad-CAM comparison (Normal vs Pneumonia)...")

    # 1. Load data
    loaders, classes = get_data_loaders(batch_size=1)
    _, _, test_loader = loaders

    # 2. Load ResNet18 model
    path_resnet = "src/Deliverable 2/models/best_resnet18.pth"
    model = timm.create_model('resnet18', pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(path_resnet, map_location=device))
    model.eval()

    # 3. Configure Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Find one Normal image (0) and one Pneumonia image (1)
    img_normal, img_pneumonia = None, None
    tensor_normal, tensor_pneumonia = None, None
    
    for images, labels in test_loader:
        label = labels[0].item()
        if label == 0 and img_normal is None:  # Normal Class
            tensor_normal = images.to(device)
            img_normal = unnormalize(images[0])
        elif label == 1 and img_pneumonia is None:  # Pneumonia Class
            tensor_pneumonia = images.to(device)
            img_pneumonia = unnormalize(images[0])
            
        if img_normal is not None and img_pneumonia is not None:
            break

    # 5. Generate heatmaps pointing to the true class of each image
    # For Normal, we look at what makes it think it's Normal (target 0)
    # For Pneumonia, we look at what makes it think it's Pneumonia (target 1)
    cam_normal = cam(input_tensor=tensor_normal, targets=[ClassifierOutputTarget(0)])[0, :]
    cam_pneumonia = cam(input_tensor=tensor_pneumonia, targets=[ClassifierOutputTarget(1)])[0, :]

    vis_normal = show_cam_on_image(img_normal, cam_normal, use_rgb=True)
    vis_pneumonia = show_cam_on_image(img_pneumonia, cam_pneumonia, use_rgb=True)

    # 6. Create the 2x2 grid for the presentation
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    # Row 1: Healthy Patient (Normal)
    ax[0, 0].imshow(img_normal)
    ax[0, 0].set_title("Original: Healthy (Normal)", fontsize=14)
    ax[0, 0].axis('off')

    ax[0, 1].imshow(vis_normal)
    ax[0, 1].set_title("Grad-CAM: Why is it Normal?", fontsize=14)
    ax[0, 1].axis('off')

    # Row 2: Sick Patient (Pneumonia)
    ax[1, 0].imshow(img_pneumonia)
    ax[1, 0].set_title("Original: Sick (Pneumonia)", fontsize=14)
    ax[1, 0].axis('off')

    ax[1, 1].imshow(vis_pneumonia)
    ax[1, 1].set_title("Grad-CAM: Why is it Pneumonia?", fontsize=14)
    ax[1, 1].axis('off')

    plt.tight_layout()
    save_path = "src/Deliverable 2/models/gradcam_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n[+] Success! Comparison image saved to: {save_path}")
    plt.show()