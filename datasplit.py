import shutil
import os
import glob

event = 'diarrhea'

# Remove old files:
img_files = glob.glob("./images/**/" + event + "/**/*.png", recursive=True)
for f in img_files:
    os.remove(f)


# Source path
src = "./" + event
src_files = os.listdir(src)
# print(src_files)

# Destination path
train_dest = "./images/train/" + event 
val_dest = "./images/validate/" + event 
test_dest = "./images/test/" + event 


def copy_file(src, percent_train, percent_test):
    file_list = os.listdir(src)
    splitTrain = int(round(percent_train * len(file_list)))
    splitTest = int(round(percent_test * len(file_list)))
    # print(splitTrain)
    copy_file.train_ls = file_list[:splitTrain]
    copy_file.test_ls = file_list[splitTrain : splitTrain + splitTest]
    copy_file.val_ls = file_list[splitTrain + splitTest :]


copy_file(src, 0.7, 0.1) # test, train, val


# Move content of source to destination, create folder if does not exist
for file in src_files:
    filename = os.path.basename(file)
    filename = os.path.splitext(filename)[0]
    if file in copy_file.train_ls:
        shutil.copy(src + "/" + file, train_dest)
    elif file in copy_file.test_ls:
        shutil.copy(src + "/" + file, test_dest)
    elif file in copy_file.val_ls:
        shutil.copy(src + "/" + file, val_dest)
