from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image = Image.open(r'uploads/lost_item.jpg')



# Load image
image_path = "static/uploads/11f69a87-4198-42ab-bede-ced880eca1e8.jpg"  # local image path
image = Image.open(image_path).convert("RGB")

# Load pretrained model & processor
model_name = "google/mobilenet_v2_1.0_224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)




inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_idx = torch.max(probs, dim=-1)

# Extract and clean the predicted label
raw_label = model.config.id2label[top_idx.item()]
clean_label = raw_label.split(",")[0].strip()  # take only the first word before comma

print(f"{clean_label}")