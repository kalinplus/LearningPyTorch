from torch.utils.tensorboard import SummaryWriter
import cv2
from PIL import Image
import numpy as np

writer = SummaryWriter("log")

image_path = "请指定昆虫数据集图片路径"
image = Image.open(image_path)
# print(type(image))
image_array = np.array(image)
# print(type(image_array))
# print(image_array.shape)

# # 根据说明文档， 输入图像需要是 tensor，numpy，string 或 blobname
writer.add_image('train', image_array, 1, dataformats='HWC')

image_path2 = "请指定昆虫数据集图片路径"
image2 = cv2.imread(image_path2)
# print(type(image2))
# print(image2.shape)
writer.add_image('train', image2, 2, dataformats='HWC')
# 记得关闭，不然可能刷新了也加载不出更新
writer.close()