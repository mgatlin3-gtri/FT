# Script fr taking an h5 file and turning it into a CNN visualization

from tkinter import font
from keras.models import load_model
import visualkeras
from PIL import ImageFont
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[GlobalAveragePooling2D]['fill'] = 'navy'
color_map[Dropout]['fill'] = 'gray'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'


model = load_model("bestCNN11.h5")
font = ImageFont.truetype("Arial Unicode.ttf", 24) # embiggen
visualkeras.layered_view(model, color_map=color_map, to_file="network.png", scale_xy=1, legend=True, font=font)
visualkeras.layered_view(model, draw_volume=False, color_map=color_map, to_file="network-flat.png", scale_xy=1, legend=True, font=font)
