import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import torch

session = ort.InferenceSession("resnet18.onnx",providers=['CPUExecutionProvider'])

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "/data/hanzt1/he/codes/engine/examples/resnet/dog.jpg")
#try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)

input_tensor = input_tensor.unsqueeze(0)

input_numpy = input_tensor.cpu().detach().numpy()
print(f'input shape = {input_numpy.shape}')
output = session.run(None, {'input': input_numpy})

def get_dimension(lst):
    dimension = 1
    while isinstance(lst, list):
        lst = lst[0] if lst else None
        dimension += 1
    return dimension - 1

print(f"output shape = {get_dimension(output)}")
output = torch.from_numpy(output[0])

probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())