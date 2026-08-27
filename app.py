from flask import Flask,render_template, request
from torchvision import transforms 
import torch
from PIL import Image
import io
from model import CifarClassifier
app=Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=CifarClassifier()
model = torch.load(r'C:\Users\yusuf\OneDrive\Documents\projects\python _projects\Image Classifier Web App\new_model.pt', map_location=device, weights_only=False)   
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
@app.route("/",  methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # handle the uploaded image
        file = request.files.get("file")
        file_content = file.read()
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        img_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            transforms.Resize((32, 32))])
        img_tensor=img_transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)                            
        if file_content:
            return class_names[predicted.item()]
        else:
            return "Uploaded Unsuccessful"
    else:
        return render_template('index.html')                                                     
if __name__ == "__main__":
    app.run(debug=True)