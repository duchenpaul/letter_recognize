import config

import os
import numpy as np

import random

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import toolkit_file

font_dir = config.FONT_DIR
char_set = config.char_set
training_data_dir = config.TRAINING_DATA_DIR


def generate_image(font_file, char):
    '''
    Example: generate_image('a.ttf', 'A', (1, 1))
    '''
    W, H = 50, 50
    size = (W, H)
    font_size = 35

    image = Image.new('1', size)
    font = ImageFont.truetype(font_file, font_size)
    draw = ImageDraw.Draw(image)

    # pos = (random.randint(0, 17), random.randint(0, 3))

    w, h = draw.textsize(char, font)
    # print(w, h)
    pos = (int((W - w) / 2), int((H - h) / 2))

    # print(pos)
    draw.text(pos, char, font=font, fill=(255))
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.convert('1')
    # image.save('test.jpg')
    return image


def batch_generate(font_file):
    basename = toolkit_file.get_basename(font_file)
    for order in char_set:
        generate_image(font_file, chr(order)).save(os.path.join(
            training_data_dir, str(order), '{}.jpg'.format(basename)))


if __name__ == '__main__':
    '''
    Sushanty
    Fixedsys500c
    Helvetica
    '''
    font_list = toolkit_file.get_file_list(font_dir)
    for fontName in font_list:
        font = '{}'.format(fontName)
        print(font)
        # char = 'd'
        # generate_image(font, char)

        # import time
        # for i in range(20):
        #     generate_image(font, char)
        #     time.sleep(.5)

        batch_generate(font)
