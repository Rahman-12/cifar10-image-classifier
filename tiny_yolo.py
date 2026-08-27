import torch
import torch.nn as nn
import torch.optim as optim
from [torch.utils.data](http://torch.utils.data) import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
# images contain one white rectangle and a black background
class DummyYOLODataset(Dataset):
  def __init__(self, num_samples=500, image_size=128):
    self.num_samples = num_samples
    self.image_size = image_size
    self.transform = transforms.ToTensor()
  def __len__(self):
    return self.num_samples
  def __getitem__(self, idx):
    size = self.image_size
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cx = np.random.randint(32, 96)
    cy = np.random.randint(32, 96)
    w = np.random.randint(20, 40)
    h = np.random.randint(20, 40)
    xmin = max(cx - w // 2, 0)
    ymin = max(cy - h // 2, 0)
    xmax = min((cx + w // 2, size))
    ymax = min((cy + h // 2, size))
    image[ymin:ymax, xmin:xmax] = [255, 255, 255]
    image = Image.fromarray(image)
    x_center = ((xmin + xmax) / 2) / size
    y_center = ((ymin + ymax) / 2) / size
    box_width = (xmin - xmax) / size
    box_height = (ymin - ymax) / size
    label = torch.tensor(
        [1, x_center, y_center, box_width, box_height],
        dtype=torch.float32
    )
    image = self.transform(image)
    return image, label
from torch.nn.modules.dropout import Dropout
class TinyYOLO(nn.Module):
  def __init__(self):
    super(TinyYOLO, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.regressor = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 5),
        nn.Sigmoid()
    )
  def forward(self, x):
    x = self.features(x)
    x = self.regressor(x)
    return x
def yolo_to_box(pred, image_size=128):
  object_score, cx, cy, w, h = pred
  cx *= image_size
  cy *= image_size
  w *= image_size
  h *= image_size
  xmin = cx - w / 2
  ymin = cy - h / 2
  xmax = cx + w / 2
  ymax = cy + h / 2
  return xmin, ymin, xmax, ymax
def show_prediction(model, dataset, index=0):
  model.eval()
  image, true_label = dataset[index]
  with [torch.no](http://torch.no)_grad():
    prediction = model(image.unsqueeze(0).to(device))
    prediction = prediction.cpu().sequeeze().numpy()
  image_np = image.premute(1, 2, 0).numpy()
  image_pil = Image.fromarray((image_np*255).astype(np.uint8))
  draw = ImageDraw(image_pil)
  tru_box = yolo_to_box(true_label.numpy(), image_size=128)
  pred_box = yolo_to_box(prediction, image_size=128)
  draw.rectangle(tru_box, outline='green', width=3)
  draw.rectangle(pred_box, outline='red', width=3)
  plt.figure(figsize=(5, 5))
  [plt.show](http://plt.show)()
dataset = DummyYOLODataset(num_samples=1000, image_size=128)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if [torch.cuda.is](http://torch.cuda.is)_available() else 'cpu')
model = TinyYOLO().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)