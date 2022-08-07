import os, shutil


operation = "delete"
categories = ["diarrhea", "farts and poop", "pee"]

if operation == "add":
    for category in categories:
        for file in os.listdir("images\\sim\\" + category):
            shutil.copy("images\\sim\\" + category + "\\" + file, "images\\train3\\" + category + "\\" + file)
elif operation == "delete":
    for category in categories:
        for file in os.listdir("images\\train3\\" + category):
            if file.find(".wav") != -1:
                os.remove("images\\train3\\" + category + "\\" + file)
