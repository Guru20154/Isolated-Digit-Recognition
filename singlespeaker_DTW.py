import os
import librosa
import numpy as np
from fastdtw import fastdtw
import sys

def extract_frames(audio_data, window_size, stride):
    num_frames = (len(audio_data) - window_size) // stride + 1
    frames = np.zeros((num_frames, window_size))

    for i in range(num_frames):
        start = i * stride
        end = start + window_size
        frames[i] = audio_data[start:end]

    return frames

def extract_mfcc_features(frames, sample_rate):
    mfcc_features = []
    for frame in frames:
        mfcc = librosa.feature.mfcc(y=frame, sr=sample_rate)
        mfcc_features.append(mfcc)
    
    return np.array(mfcc_features)

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

def evaluate_accuracy(results):
    count = 0
    for result in results:
        split1 = result[0].rsplit("_", 2)[0]
        split2 = result[1].rsplit("_", 2)[0]
        if split1 == split2:
            count += 1
    return count

def main():
    name = 'jackson'
    train_drec = 'personwise_data'+ '/' + name + '/' +'train'
    test_drec = 'personwise_data'+ '/' + name + '/' +'test'
    window_size = 240
    stride = 120
    sample_rate = 22050

    A_train = []
    label_name_train = []
    for filename in os.listdir(train_drec):
        path = train_drec + "/" + filename
        audio_data, sample_rate = librosa.load(path)
        frames = extract_frames(audio_data, window_size, stride)
        A_train.append(frames)
        label_name_train.append(filename.rsplit(".", 1)[0])

    A_test = []
    label_name_test = []
    for filename in os.listdir(test_drec):
        path = os.path.join(test_drec, filename)
        audio_data, _ = librosa.load(path, sr=sample_rate)
        frames = extract_frames(audio_data, window_size, stride)
        A_test.append(frames)
        label_name_test.append(filename.rsplit(".", 1)[0])

    mfcc_train = extract_mfcc_features(np.array(A_train), sample_rate)
    mfcc_test = extract_mfcc_features(np.array(A_test), sample_rate)

    distances,result = calculate_dtw_distance(mfcc_test, mfcc_train)

    R = []
    j=0
    for distance, index in result:
        R.append([label_name_train[int(index)], label_name_test[j]])
        j=j+1;

    accuracy = evaluate_accuracy(R)
    print("Accuracy:", accuracy)
    print(accuracy/j)
if __name__ == '__main__':
    main()
