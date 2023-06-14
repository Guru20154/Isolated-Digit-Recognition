#importing libraries
import os
import librosa
import numpy as np
from fastdtw import fastdtw
import sys
import matplotlib.pyplot as plt

#Calculate the number of frames based on window size and stride
def extract_frames(audio_data, window_size, stride):
    num_frames = (len(audio_data) - window_size) // stride + 1
    frames = np.zeros((num_frames, window_size))

    for i in range(num_frames):
        start = i * stride
        end = start + window_size
        frames[i] = audio_data[start:end]

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


#calculate mfcc 
def extract_mfcc_features(frames, sample_rate):
    mfcc_features = []
    for frame in frames:
        mfcc = librosa.feature.mfcc(y=frame, sr=sample_rate)
        mfcc_features.append(mfcc)
    
    return np.array(mfcc_features)

#calculate dtw distances
def calculate_dtw_distance(test_features, train_features):
    result = []
    distances=[]
    for j in range(50):
        min_distance = sys.maxsize
        min_index = 0
        dis = []
        for i in range(450):
            distance, _ = fastdtw(test_features[j], train_features[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
            dis.append(distance)
        result.append((min_distance, min_index))
        distances.append(dis)

    return np.array(distances),np.array(result)

#calcualting accuracy
def evaluate_accuracy(results):
    count = 0
    for result in results:
        split1 = result[0].rsplit("_", 2)[0]
        split2 = result[1].rsplit("_", 2)[0]
        if split1 == split2:
            count += 1
    return count

def main():
    name = 'jackson'#name of speaker
    train_drec = 'personwise_data'+ '/' + name + '/' +'train'#train and test drec
    test_drec = 'personwise_data'+ '/' + name + '/' +'test'
    window_size = 240#window size of framing
    stride = 120#overlapping of frmaes
    sample_rate = 22050#sample rate

    #extracting frames for each train and test file
    A_train ,label_name_train,no_label = frame_retrieve(train_drec,window_size,stride)
    A_test ,label_name_test = frame_retrieve(test_drec,window_size,stride)

    #mfcc features
    mfcc_train = extract_mfcc_features(np.array(A_train), sample_rate)
    mfcc_test = extract_mfcc_features(np.array(A_test), sample_rate)

    #dtw distance
    distances,result = calculate_dtw_distance(mfcc_test, mfcc_train)

    R = []
    j=0
    for distance, index in result:
        R.append([label_name_train[int(index)], label_name_test[j]])
        j=j+1;

    accuracy = evaluate_accuracy(R)
    print("Accuracy:", accuracy/j)

    #plotting distances for each number
    plt.bar(no_label,distances[0],color = 'maroon')
    plt.xlabel("Speaker")
    plt.ylabel("DTW distances")
    plt.title("Distances of " + str(label_name_test[0]) + " with test set")
    plt.show()


if __name__ == '__main__':
    main()
