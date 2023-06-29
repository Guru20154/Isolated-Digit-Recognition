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
    A = [[[0 for _ in range(240)]] for _ in range(10)] #list to store enframes of each audio file
    label_name = [] #list containing the number and name of speaker
    label=[]  #list containing number spoken in audio file
    test_speaker=["nicolas","theo","yweweler"]
    train_speaker=["george","jackson","lucas"]
    for filename in os.listdir(drec):
        if filename.rsplit("_", 2)[1] in train_speaker:
            path = drec + "/" + filename
            audio_data, sample_rate = librosa.load(path)  #loading audio fike
            frames = extract_frames(audio_data, window_size, stride)   #function to extract frames
            label_name.append(filename.rsplit(".", 1)[0])
            file_class = filename.rsplit("_", 2)[0]
            label.append(file_class)
            # Determine the maximum number of rows and columns between the two lists
            max_rows = max(len(A[int(file_class)]), len(frames))
            max_cols = 240#max(len(row) for row in A[int(file_class)]+frames)

            r=[]
            # Iterate over the rows
            for row in range(max_rows):
                temp = []

                # Iterate over the columns within each row
                for col in range(max_cols):
                    # Add the corresponding elements if available, or use 0 otherwise
                    element1 = A[int(file_class)][row][col] if row < len(A[int(file_class)]) and col < len(A[int(file_class)][row]) else 0
                    element2 = frames[row][col] if row < len(frames) and col < len(frames[row]) else 0
                    temp.append(element1 + element2)
                r.append(temp)
            A[int(file_class)]=r
    
    result=[]
    # Iterate over the elements of the nested list
    for sublist1 in A:
        temp1 = []
        for sublist2 in sublist1:
            temp2 = [int(element / 135) for element in sublist2]
            temp1.append(temp2)
        result.append(temp1)
    return result, label_name,label

def test_frame_retrieve(drec, window_size, stride):
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
        mfcc = librosa.feature.mfcc(y=np.asarray(frame, dtype=np.float32), sr=sample_rate)  #creating mfcc features
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

#function to retrieve labels of least acquired distance samples
def result_label(lable_name_test,result):
    R = [] #list to contain label of train and test audio file with least distance
    j=0
    for distance, index in result:
        R.append([index, lable_name_test[j]])#index contain index of train data with least distnace
        j=j+1
    return R

#function to calculate accuracy
def evaluate_accuracy(results):
    count = 0
    for result in results:
        split1 = result[0]#obtaining number spoken in each audio file
        split2 = int(result[1].rsplit("_", 2)[0])#(e.g:-result(0_george_2,0_george_32))split1 = 0, split2 = 0
        if split1 == split2:
            count += 1
    return count

#george,jackson,lucas
#nicolas,theo,yweweler
def main():
    train_drec = 'training-spectrograms'#directory of train data
    test_drec1 = 'personwise_data'+ '/' + 'nicolas' + '/' +'test'#directory of test data
    test_drec2 = 'personwise_data'+ '/' + 'theo' + '/' +'test'#directory of test data
    test_drec3 = 'personwise_data'+ '/' + 'yweweler' + '/' +'test'#directory of test data
    window_size = 240 #window size of frame
    stride = 120 #overlaping of frames
    sample_rate = 22050 #sample rate

    A_train,label_name_train,no_label = frame_retrieve(train_drec, window_size, stride)#obtaining enframed train files
    nicolas_test,nicolas_label_name_test,_ = test_frame_retrieve(test_drec1, window_size, stride)##obtaining enframed test files
    theo_test,theo_label_name_test,_ = test_frame_retrieve(test_drec2, window_size, stride)##obtaining enframed test files
    yweweler_test,yweweler_label_name_test,_ = test_frame_retrieve(test_drec3, window_size, stride)##obtaining enframed test files
    
    print(len(A_train))
    print(len(A_train[0]))
    print(type(A_train))
    mfcc_train = extract_mfcc_features(np.array(A_train), sample_rate)#obtaining mfcc train files
    nicolas_mfcc_test = extract_mfcc_features(np.array(nicolas_test), sample_rate)#obtaining mfcc test files
    theo_mfcc_test = extract_mfcc_features(np.array(theo_test), sample_rate)#obtaining mfcc test files
    yweweler_mfcc_test = extract_mfcc_features(np.array(yweweler_test), sample_rate)#obtaining mfcc test files

    nicolas_distances,nicolas_result = calculate_dtw_distance(nicolas_mfcc_test, mfcc_train)#calcultation dtw distance
    theo_distances,theo_result = calculate_dtw_distance(theo_mfcc_test, mfcc_train)#calcultation dtw distance
    yweweler_distances,yweweler_result = calculate_dtw_distance(yweweler_mfcc_test, mfcc_train)#calcultation dtw distance

    R_nicolas = result_label(nicolas_label_name_test,nicolas_result) #list to contain label of train and test audio file with least distance
    accuracy = evaluate_accuracy(R_nicolas)
    print("Accuracy:", accuracy/len(R_nicolas))#accuracy
    Accuracy=[]
    Accuracy.append(accuracy)

    R_theo = result_label(theo_label_name_test,theo_result) #list to contain label of train and test audio file with least distance
    accuracy = evaluate_accuracy(R_theo)
    print("Accuracy:", accuracy/len(R_theo))#accuracy
    Accuracy.append(accuracy)

    R_yweweler = result_label(yweweler_label_name_test,yweweler_result) #list to contain label of train and test audio file with least distance
    accuracy = evaluate_accuracy(R_yweweler)
    print("Accuracy:", accuracy/len(R_yweweler))#accuracy
    Accuracy.append(accuracy)
    x=[0,1,2,3,4,5,6,7,8,9]
#plotting distances for each number
    fig = plt.figure(figsize =(10, 7))
    plt.bar(x,nicolas_distances[0],color = 'maroon')
    plt.xlabel("Speaker")
    plt.ylabel("DTW distances")
    plt.title("Distances of " + str(nicolas_label_name_test[0]) + " with test set")
    plt.show()
    
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(Accuracy)
    plt.show()

    
if __name__ == '__main__':
    main()
