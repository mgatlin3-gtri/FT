import matplotlib.pyplot as plt
import json
import wandb
from keras.models import load_model

# file to save model weights and structure to
model_file = 'bestCNN14.h5'

# load model history
history = json.load(open('model_history.json', 'r'))


# create plots of loss on training and validation data
print("Creating plots")
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Loss vs. Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training loss','validation loss'], loc='upper left')
# plt.savefig("loss.eps", format='eps')
plt.savefig('loss.png')