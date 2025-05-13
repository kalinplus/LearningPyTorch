from torchvision import transforms
from PIL import Image

totensor = transforms.ToTensor()

image_path = "请指定昆虫数据集路径"
image = Image.open(image_path)

image_tensor = totensor(image)
print(type(image_tensor))
print(image_tensor.shape)