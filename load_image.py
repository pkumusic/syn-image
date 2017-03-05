# Project for 11777
import matplotlib as plot
from PIL import Image

def load_image(path):
    im = Image.open(path).convert('RGB')
    #im.show()
    return im

def load_labels(path):
    with open(path,'r') as f:
        info = f.readline()
        labels = info.strip().split(',')
        return labels


if __name__ == "__main__":
    data_folder = 'data/102flowers/'
    # file name format: image_00000.jpg
    #load_image(data_folder + 'jpg/image_00001.jpg')
    load_labels(data_folder + 'imagelabels.txt')
