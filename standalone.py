#importing libraries   
import os
import librosa
import numpy as np
from fastdtw import fastdtw
import sys
import matplotlib.pyplot as plt

#function for slicing of audio
def extract_frames(audio_data, window_size, stride):
    num_frames = (len(audio_data) - window_size) // stride + 1 #number of frames for the audio file
    frames = np.zeros((num_frames, window_size)) #creating array to store the frames

    for i in range(num_frames): 
        start = i * stride   
        end = start + window_size
        frames[i] = audio_data[start:end]    #slicing audio file based on frame's start and end point

    return frames


#function for enframing of audio
def frame_retrieve_train(drec, window_size, stride):
    A = [] #list to store enframes of each audio file
    label_name = [] #list containing the number and name of speaker
    label=[]  #list containing number spoken in audio file
    for filename in os.listdir(drec):
        path = drec + "/" + filename
        audio_data, sample_rate = librosa.load(path)  #loading audio fike
        frames = extract_frames(audio_data, window_size, stride)   #function to extract frames
        A.append(frames)
        label_name.append(filename.rsplit(".", 1)[0])
        label.append(filename.rsplit("_", 2)[0])
    return A, label_name,label

#function to get system arguments and read the audio files
def frame_retrieve_test(sample_rate, window_size, stride):
    A_test=[]
    for arg in sys.argv[1:]: #interating over arguments
        audio_data, _ = librosa.load(arg, sr=sample_rate)
        frames = extract_frames(audio_data, window_size, stride)
        A_test.append(frames)
    return A_test

#function to obtain mfcc features
def extract_mfcc_features(frames, sample_rate):
    mfcc_features = []
    for frame in frames:
        mfcc = librosa.feature.mfcc(y=frame, sr=sample_rate)  #creating mfcc features
        mfcc_features.append(mfcc)
    
    return np.array(mfcc_features)

#function to get dtw distance
def calculate_dtw_distance(test_features, train_features):
    result = [] #list minimum distance and index of the train feature to obtain label
    distances=[] #list for distance ffrom each train data for each test data
    for test_feature in test_features: #loop for each test audio
        min_distance = sys.maxsize 
        min_index = 0
        dis = []  #lsit to contain the distance for current test audio
        for i, train_feature in enumerate(train_features):
            distance, _ = fastdtw(test_feature, train_feature) #calculating dtw distance
            if distance < min_distance:
                min_distance = distance #minimun distance
                min_index = i #index of that minimum distance train audio file
            dis.append(distance)
        result.append((min_distance, min_index))
        distances.append(dis)

    return np.array(distances),np.array(result)

def main():
    train_drec = 'training-spectrograms' #training directory
    window_size = 240 #window size for framing
    stride = 120 #overlapping of frames
    sample_rate = 22050 #sample rate 

    A_train,label_name_train,label_no = frame_retrieve_train(train_drec, window_size, stride) #obtaining lable and enframes
    A_test = frame_retrieve_test(sample_rate, window_size, stride) #obtainig test enframes
    
    mfcc_train = extract_mfcc_features(np.array(A_train), sample_rate)#mfcc 
    mfcc_test = extract_mfcc_features(np.array(A_test), sample_rate)#mfcc 

    distances,result = calculate_dtw_distance(mfcc_test, mfcc_train)#dtw distance

    R = []#list to contain label of train and test audio file with least distance
    for distance, index in result:
        R.append(label_name_train[int(index)])
    for i in R:
        print(i)
    
    #plotting distances for each number
    for i in range(len(distances)):
        plt.bar(label_no,distances[i],color = 'maroon')
        plt.xlabel("Speaker")
        plt.ylabel("DTW distances")
        plt.title("Distances of " + str(sys.argv[i]) + " with test set")
        plt.show()


if __name__ == '__main__':
    main()
