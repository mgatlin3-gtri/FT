#script to add any new noise machine data from Box to simulated data folders
import os, shutil

dir = "//Users/mgatlin3/Library/CloudStorage/Box-Box/Feces Thesis/noiseMachineNoise/Recordings"
for file in os.listdir(dir):
    _index = file.find("_")
    code = file[_index + 1 : _index + 5]
    
    category = "other"
    if code == "1000":
        category = "diarrhea"
    elif code == "0100":
        category = "farts"
    elif code == "0010":
        category = "poop"
    elif code == "0001":
        category = "pee"
    sub_dir = "audio/simulated/" + category

    if not os.path.isfile(sub_dir + "/" + file):
        shutil.copy(dir + "/" + file, sub_dir + "/" + file)
        print("Added " + file + " to " + category)
