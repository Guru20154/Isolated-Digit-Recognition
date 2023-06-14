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
def frame_retrieve(drec, window_size, stride):
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

#function to calculate accuracy
def evaluate_accuracy(results):
    count = 0
    for result in results:
        split1 = result[0].rsplit("_", 2)[0]#obtaining number spoken in each audio file
        split2 = result[1].rsplit("_", 2)[0]#(e.g:-result(0_george_2,0_george_32))split1 = 0, split2 = 0
        if split1 == split2:
            count += 1
    return count

def main():
    train_drec = 'personwise_data'+ '/' + 'jackson' + '/' +'train'#directory of train data
    test_drec = 'personwise_data'+ '/' + 'george' + '/' +'test'#directory of test data
    window_size = 240 #window size of frame
    stride = 120 #overlaping of frames
    sample_rate = 22050 #sample rate

    A_train,label_name_train,no_label = frame_retrieve(train_drec, window_size, stride)#obtaining enframed train files
    A_test,label_name_test,_ = frame_retrieve(test_drec, window_size, stride)##obtaining enframed test files
    
    mfcc_train = extract_mfcc_features(np.array(A_train), sample_rate)#obtaining mfcc train files
    mfcc_test = extract_mfcc_features(np.array(A_test), sample_rate)#obtaining mfcc test files

    distances,result = calculate_dtw_distance(mfcc_test, mfcc_train)#calcultation dtw distance

    R = [] #list to contain label of train and test audio file with least distance
    j=0
    for distance, index in result:
        R.append([label_name_train[int(index)], label_name_test[j]])#index contain index of train data with least distnace
        j=j+1
    accuracy = evaluate_accuracy(R)
    print("Accuracy:", accuracy/j)#accuracy

    #plotting distances for each number
    plt.bar(no_label,distances[0],color = 'maroon')
    plt.xlabel("Speaker")
    plt.ylabel("DTW distances")
    plt.title("Distances of " + str(label_name_test[0]) + " with test set")
    plt.show()

    
if __name__ == '__main__':
    main()
