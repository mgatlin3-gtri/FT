# **The Feces Thesis: Using ML to Diagnose Various Toilet Audio**

This repository uses Keras to generate neural networks which can classify spectrograms, visual representations of audio files. Audio files may be taken from the internet as mp3's and used to generate spectrograms. Spectrograms can be augmented for better ML results.

While most of the scripts related to manipulating audio and image files were run on a Windows machine, the programs which create and make predictions using neural networks were run on Linux OS. Thus, you may find some inconsistencies among the files, primarily the switch between using '/' and '\\' to indicate sub-directories.

# Most Important Files

bestCNN11.h5: best model as of 21 July 2022

2dCNN.py: used for training and saving a CNN classifier model and create loss plot during training

testCNN.py: used to make predictions from an existing CNN model and generate confusion matrix figures

full_spectros.py: used to generate spectrograms from a given segment of an audio file

prepared_spectros.py: used to generate a full set of spectrograms from existing spectrograms in the 'originals directory. (A good combination is to use full_spectros to put spectrograms in originals, then go through the directory and delete any that are not useful, then run prepared_spectros)

count_data.py: used to tally the number of spectrograms in various locations

gen_spect4fig.py: used to generate a spectrograms for comparison figure between online and simulated data

plot_spect4fig.py: used to create a plot of spectrograms for comparison figure between online and simulated data

visualize_model.py: used to generate model architecture figure for journal
