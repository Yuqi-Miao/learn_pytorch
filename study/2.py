from PIL import Image

# 打开图片文件
image = Image.open("./data/train/ants/0013035.jpg")  # 替换为您的图片文件路径

# 显示图片信息
print("图片格式:", image.format)
print("图片大小:", image.size)
print("图片模式:", image.mode)

# 显示图片
image.show()
