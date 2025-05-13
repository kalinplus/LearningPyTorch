from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

image_path = "../dataset/test1.webp"
image = Image.open(image_path)

# 创建类对象，trnasforms 只是一个工具包
totensor = transforms.ToTensor()
image_tensor = totensor(image)
writer.add_image("ToTensor", image_tensor, 0)

# ToPILImage
topilimage = transforms.ToPILImage()
image_pil = topilimage(image_tensor)
# image_pil.show()

# Normalize [0, 1] -> [-1, 1]
# Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n`` channels
# ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
image_tensor_normalize = normalize(image_tensor) # ToTensor 自动转换为 CHW 格式
writer.add_image("Normalize", image_tensor_normalize, 0)

# Resize
# 进行缩放，接收参数为 int 或者 tuple；其中 int 缩放短边，并保持原始比例
resize = transforms.Resize((1024, 1024))
image_resize = resize(image)
image_resize = totensor(image_resize)
writer.add_image("Resize", image_resize, 0)

# Compose
# Compose 接收的输入为一个列表，其中每个元素都是一个 transforms 中的类对象
compose = transforms.Compose([transforms.Resize(1024), totensor])
image_resize_tensor = compose(image)
writer.add_image("Compose", image_resize_tensor, 0)

# RandomCrop 随机裁下输入图片的指定大小
# 接收 PIL 和 tensor 作为输入 （文档似乎没有提到PIL）
randomcrop = transforms.RandomCrop(512)
image_crop = randomcrop(image)
# print(type(image_crop))
# print(image_crop.size)
for i in range(10):
    image_crop_tensor = randomcrop(image_tensor)
    writer.add_image("RandomCrop", image_crop_tensor, i)

writer.close()
