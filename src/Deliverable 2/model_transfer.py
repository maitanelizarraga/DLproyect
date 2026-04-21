import torch
import torch.nn as nn
from torchvision import models

def get_transfer_model(num_classes=2):
    # we download the pretrained VGG16 model, that is a trained model on ImageNet, which has learned to extract useful features from images like borders, textures, shapes, etc. 
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # We freezw the parameters of the pretrained model, so it will no change its working
    for param in model.parameters():
        param.requires_grad = False

    # 3. Adaptar la arquitectura (Classifier)
    # VGG16 termina en un bloque 'classifier'. Vamos a reemplazar 
    # la última capa lineal para que coincida con nuestras clases.
    
    # Obtenemos el número de neuronas de entrada de la última capa
    num_ftrs = model.classifier[6].in_features
    
    # Reemplazamos la capa 6 por una nueva que SÍ tendrá requires_grad = True por defecto
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model

# --- Configuración de Entrenamiento ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_transfer_model(num_classes=2).to(device)

# Solo optimizamos los parámetros de la capa que NO está congelada
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)