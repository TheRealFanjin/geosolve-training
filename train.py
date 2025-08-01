import torch
from torch import nn
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import RN50Model
from image_loader import CustomImageDataset

INTERNAL_DATA = "dataset/train"
MODEL_SAVE_PATH = "saved_models/"
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', DEVICE)

# Input transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load data
print('loading data')
train_ds = CustomImageDataset(root=INTERNAL_DATA, transform=transform)
class_to_idx = train_ds.class_to_idx
print(class_to_idx)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)

# Load model
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
base_model = RN50Model(resnet).to(DEVICE)
print(base_model)
input('Press enter to continue')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
torch.backends.cudnn.benchmark = True
compiled_model = torch.compile(base_model, mode="max-autotune")
scaler = torch.amp.GradScaler('cuda')

print('training')
loss_history = []
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    compiled_model.train()

    for images, labels in tqdm(train_dl,
                               desc=f"Epoch {epoch}/{EPOCHS}",
                               leave=False,
                               ncols=80):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = compiled_model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        epoch_loss += loss.item() * batch_size

    ckpt = f"{MODEL_SAVE_PATH}/model_e{epoch}.pt"
    torch.save({
        'state_dict': base_model.state_dict(),
        'class_to_idx': class_to_idx,
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'loss_history': loss_history
    }, ckpt)
    print(f"saved checkpoint {ckpt}\n")

    epoch_loss /= len(train_ds)
    loss_history.append(epoch_loss)
    scheduler.step(epoch_loss)
    print(f"Epoch {epoch:02d}/{EPOCHS} — loss: {epoch_loss:.4f}\n")

final_dir = os.path.join(MODEL_SAVE_PATH, "final_trained_model")
os.makedirs(final_dir, exist_ok=True)
ckpt = f"final.pt"
torch.save({
    'state_dict': base_model.state_dict(),
    'class_to_idx': class_to_idx,
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'loss_history': loss_history
}, ckpt)
print(f"saved checkpoint {ckpt}")
print(f"saved final model to {final_dir}")

epochsRange = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochsRange, loss_history, label="Training loss", color="blue")
plt.title("Training Over Loss Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (Cross Entropy)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
