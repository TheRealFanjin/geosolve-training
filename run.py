from model import Network
import torch
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(DEVICE)
IMG_SIZE = 256
checkpoint = torch.load('saved_models/model_batch_180000.pt', map_location=DEVICE)
# checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint['state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

img = Image.open("/home/fanjinm/Documents/GeoSolve/python-geosolve/dataset/train/TUR/__cGjXyCuVzmtnBCXcamFg_Turkey_x0_y1.jpeg").convert("RGB")
batch = transform(img).unsqueeze(0).to(DEVICE)  # shape (1,3,H,W)

with torch.no_grad():  # no grads needed
    logits = model(batch)  # (1, num_classes)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()  # integer index
cs = {v: k for k, v in {'ALB': 0, 'AND': 1, 'ARE': 2, 'ARG': 3, 'AUS': 4, 'AUT': 5, 'BEL': 6, 'BGD': 7, 'BGR': 8, 'BOL': 9, 'BRA': 10, 'BTN': 11, 'BWA': 12, 'CAN': 13, 'CHE': 14, 'CHL': 15, 'COL': 16, 'CZE': 17, 'DEU': 18, 'DNK': 19, 'DOM': 20, 'ECU': 21, 'ESP': 22, 'EST': 23, 'FIN': 24, 'FRA': 25, 'GBR': 26, 'GRC': 27, 'GRL': 28, 'GTM': 29, 'HRV': 30, 'HUN': 31, 'IDN': 32, 'IND': 33, 'IRL': 34, 'ISL': 35, 'ISR': 36, 'ITA': 37, 'JOR': 38, 'JPN': 39, 'KAZ': 40, 'KEN': 41, 'KGZ': 42, 'KHM': 43, 'KOR': 44, 'LAO': 45, 'LBN': 46, 'LKA': 47, 'LSO': 48, 'LTU': 49, 'LUX': 50, 'LVA': 51, 'MCO': 52, 'MDA': 53, 'MEX': 54, 'MKD': 55, 'MLT': 56, 'MNE': 57, 'MNG': 58, 'MYS': 59, 'NGA': 60, 'NLD': 61, 'NOR': 62, 'NZL': 63, 'PAN': 64, 'PER': 65, 'PHL': 66, 'POL': 67, 'PRT': 68, 'QAT': 69, 'ROU': 70, 'RUS': 71, 'RWA': 72, 'SEN': 73, 'SGP': 74, 'SMR': 75, 'SRB': 76, 'STP': 77, 'SVK': 78, 'SVN': 79, 'SWE': 80, 'SWZ': 81, 'THA': 82, 'TUN': 83, 'TUR': 84, 'TWN': 85, 'TZA': 86, 'UGA': 87, 'UKR': 88, 'URY': 89, 'USA': 90, 'VNM': 91, 'VUT': 92, 'ZAF': 93}.items()}
d = {cs[count]: round(i, 5) for count, i in enumerate(probs.tolist()[0])}
print(f'Prediction: {cs[pred]} {d[cs[pred]]}')
for k, v in d.items():
    print(k, v)
