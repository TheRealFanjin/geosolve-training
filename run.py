from model import RN50Model
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision import transforms
from PIL import Image
from image_loader import CustomImageDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
model = RN50Model(resnet).to(DEVICE)
IMG_SIZE = 256
checkpoint = torch.load('saved_models/model_batch_20000.pt', map_location=DEVICE)
# checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint['state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

data = CustomImageDataset('dataset/val', transform=transform)
data = DataLoader(data, batch_size=1, shuffle=False)

for images, _ in data:
    # (1, N, 3, 224, 224)
    images = images.to(DEVICE)
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()

cs = {v: k for k, v in checkpoint['class_to_idx'].items()}
d = {cs[count]: round(i, 5) for count, i in enumerate(probs.tolist()[0])}
print(f'Prediction: {cs[pred]} {d[cs[pred]]}')
for k, v in d.items():
    print(k, v)
