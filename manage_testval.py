# Script for adding unaugmented spectrograms in the test and validate sections to easy_test and easy_val

import os, shutil

val_dir = "images\\validate"
test_dir = "images\\test"
categories = ["farts", "poop"]

def do_it():
    for category in categories:
        for file in os.listdir(val_dir + "\\" + category):
            name = file
            png_index = name.find(".png")
            name = name[:png_index]

            num = int(name[-1])
            if (num == 0):
                shutil.copy(val_dir + "\\" + category + "\\" + file, "images\\easy_val\\" + category + "\\" + file)
        for file in os.listdir(test_dir + "\\" + category):
            name = file
            png_index = name.find(".png")
            name = name[:png_index]

            num = int(name[-1])
            if (num == 0):
                shutil.copy(test_dir + "\\" + category + "\\" + file, "images\\easy_test\\" + category + "\\" + file)
