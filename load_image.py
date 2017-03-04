# Project for 11777
import matplotlib as plot
from PIL import Image

def load_image(path):
    im = Image.open(path).convert('RGB')
    #im.show()
    return im


if __name__ == "__main__":
    data_folder = 'data/102flowers/jpg/'
    # file name format: image_00000.jpg
    load_image(data_folder + 'image_00001.jpg')
