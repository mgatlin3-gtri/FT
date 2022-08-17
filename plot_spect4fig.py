# create spectrogram comparison figure using spectrograms generated in gen_spect4fig.py
import cv2
from matplotlib import pyplot as plt
import seaborn

# create figure
fig = plt.figure(figsize=(4,2), constrained_layout=True)
  
# setting values to rows and column variables
r = 2
c = 4

base = './specFig_imgs/'
# reading images 's'=sim, 'o'=online
odia = cv2.imread(base + 'online/diarrhea/diarrhea_10hrs.mp3_210-220sec_spec0.png')
sdia = cv2.imread(base + 'simulated/diarrhea/242_1000_0_Jul-07-2022_16-01-23.wav_0-10sec_spec0.png')
opoo = cv2.imread(base + 'online/poop/audio_defecation_266483-having_a_poo_02.mp3_0-10sec_spec0.png')
spoo = cv2.imread(base + 'simulated/poop/327_0010_0_Jul-14-2022_11-31-49.wav_0-10sec_spec0.png')
opee = cv2.imread(base + 'online/pee/pee_2hrs.mp3_540-550sec_spec0.png')
spee = cv2.imread(base + 'simulated/pee/313_0001_0_Jul-14-2022_11-13-46.wav_0-10sec_spec0.png')
ofar = cv2.imread(base + 'online/farts/99_farts.mp3_410-420sec_spec0.png')
sfar = cv2.imread(base + 'simulated/farts/328_0100_0_Jul-14-2022_11-39-17.wav_0-10sec_spec0.png')

# Online plots:
# (1,1)
fig.add_subplot(r, c, 1)
plt.imshow(cv2.cvtColor(opoo, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Online", rotation='vertical', x=-0.05,y=0.3, fontsize=2)
# (1,2)
fig.add_subplot(r, c, 2)
plt.imshow(cv2.cvtColor(odia, cv2.COLOR_BGR2RGB))
plt.axis('off')
# (1,3)
fig.add_subplot(r, c, 3)
plt.imshow(cv2.cvtColor(opee, cv2.COLOR_BGR2RGB))
plt.axis('off')
# (1,4)
fig.add_subplot(r, c, 4)
plt.imshow(cv2.cvtColor(ofar, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Simulated plots:
# (2,1)
fig.add_subplot(r, c, 5)
plt.imshow(cv2.cvtColor(spoo, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Simulated", rotation='vertical', x=-0.05,y=0.3, fontsize=2)
# (2,2)
fig.add_subplot(r, c, 6)
plt.imshow(cv2.cvtColor(sdia, cv2.COLOR_BGR2RGB))
plt.axis('off')
# (2,3)
fig.add_subplot(r, c, 7)
plt.imshow(cv2.cvtColor(spee, cv2.COLOR_BGR2RGB))
plt.axis('off')
# (2,4)
fig.add_subplot(r, c, 8)
plt.imshow(cv2.cvtColor(sfar, cv2.COLOR_BGR2RGB))
plt.axis('off')
# save:
plt.savefig('spect_compare.png', dpi=600)