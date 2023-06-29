import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from PIL import Image
import os

def spectrograms(source,destination):
    j=0
    for filename in os.listdir(source):
        src = source + "/" + filename
        des = destination + "/" + filename.rsplit("_", 2)[0] + "/" + filename.rsplit(".", 1)[0] + "_spectogram_" + str(j) +".jpeg"
        target_sr = 22050  # Set your desired target sample rate
        input_size = (224, 224)  # Desired input size for the VGG model

        audio, sr = librosa.load(src)
        audio_resampled = librosa.resample(y=audio,orig_sr= sr, target_sr=target_sr)

        spectrogram = np.abs(librosa.stft(audio_resampled))

        plt.figure(figsize=(12, 6))
        ax = plt.axes(frameon=False)
        ax.set_axis_off()
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                                 sr=target_sr, x_axis='time', y_axis='log')

        # Save the figure as an image file without displaying it
        plt.savefig(des,bbox_inches='tight', pad_inches=0)
        plt.close()
        j=j+1
    
def main():
    train_drec = 'spectograms/image_train_spectrograms'
    test_drec = 'spectograms/image_test_spectrograms'
    train_src = 'training-spectrograms'
    test_src = 'testing-spectrograms'
    spectrograms(train_src,train_drec)
    spectrograms(test_src,test_drec)
    
if __name__ == '__main__':
    main()