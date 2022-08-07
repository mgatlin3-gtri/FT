# Program for counting the number of files in a given directory
# Used mainly for counting spectrograms in various categories of 'images'

import os
import os.path

type = "images"
sections = ["train", "validate", "test"]
categories = ["diarrhea", "farts", "pee", "poop"]

print("List of all data types/quantity in images directory:")

# uses os.listdir to loop through files
grand_total = 0
for section in sections:
    print("\n" + section[0].upper() + section[1:] + ":")
    total = 0
    for category in categories:
        print("\t" + category + ": " + str(len([name for name in os.listdir(type + "\\" + section + "\\" + category)])))
        total += len([name for name in os.listdir(type + "\\" + section + "\\" + category)])
    print("\ttotal: " + str(total))
    grand_total += total
print("\nGrand Total: " + str(grand_total))
