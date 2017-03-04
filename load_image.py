# Project for 11777
import matplotlib as plot
from scipy import misc

def load_image(path):
    image = misc.imread(path)
    print image


if __name__ == "__main__":
    data_folder = 'data/102flowers/jpg/'
    # file name format: image_00000.jpg
    load_image(data_folder + 'image_00001.jpg')
