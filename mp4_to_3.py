# Script for converting mp4 to mp3 files, used mainly for processing full_david_meyer date

from moviepy.editor import *
import os

dir = "C:\\Users\\apopa6\\Box\\Feces Thesis\\audio\\OneDrive_2022-07-11\\Trimmed Front Video Sequences"
for mp4_file in os.listdir(dir):
    if int(mp4_file[3 : -5]) > 3:
        videoClip = VideoFileClip(dir + "\\" + mp4_file)
        audioClip = videoClip.audio
        
        base, ext = os.path.splitext(mp4_file)
        print(base)
        audioClip.write_audiofile("images\\full_david_meyer\\diarrhea\\" + base + ".mp3")

        audioClip.close()
        videoClip.close()
