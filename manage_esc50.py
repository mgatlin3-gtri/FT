# Assuming the ESC-50 dataset is stored locally, this script prunes the dataset so that undesirable background noise
# (such as flush and water pouring sounds) are removed

import os

dir = "C:\\Users\\Anthony Popa\\ESC-50\\audio"

for file in os.listdir(dir):
    name = file
    dash_index = name.find("-")
    while dash_index != -1:
        name = name[dash_index+1 : ]
        dash_index = name.find("-")

    end_index = name.find(".wav")
    name = name[ : end_index]

    num = int(name)
    if (num == 15) or (num == 17) or (num == 18):
        print("Removed " + file)
        os.remove(dir + "\\" + file)
