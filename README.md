# Isolated Digit Recognition with DTW

**Name**: Gurkanwal Singh  
**Mentor**: Prof. Padmanabhan Rajan  

## Introduction

The objective of this report is to explore the efficacy of Dynamic Time Warping (DTW) for isolated digit recognition. DTW is a time series alignment algorithm that can handle variations in speech patterns and background noise, making it a promising candidate for accurate digit identification.

## DTW

Dynamic Time Warping (DTW) is a powerful algorithm used for comparing and aligning time series data, particularly useful in speech and pattern recognition tasks. It measures the similarity between two sequences by warping the time axis to find the optimal alignment, allowing efficient comparison of sequences with varying lengths and temporal distortions. DTW has applications in various fields, including speech recognition, gesture recognition, and bioinformatics, where traditional distance metrics may fail due to temporal variations. [1]

## Experiments

### Audio Representations Used:
The dataset comprised isolated digit utterances in `.wav` format. Preprocessing involved frame-level extraction from the audio files and feature extraction using Mel-Frequency Cepstral Coefficients (MFCCs). MFCCs were chosen as the audio representation due to their ability to effectively capture relevant features of the speech signal, making them suitable for our isolated digit recognition task.

### Experiment Setup:
We obtained our data from a famous dataset on GitHub known as the free-spoken digit dataset, which contains a diverse dataset of isolated digit utterances from multiple speakers. The speakers include George, Jackson, Lucas, Nicolas, Theo, and Yweweler. The dataset was divided into two subsets: one where the training and testing data belonged to each speaker, and a class-wise case, where the training and testing data came from different classes.

### Matched Case:
Matched cases include training and testing datasets from the same speakers. Two approaches were used:
1. Finding DTW distances from each individual train sample with the test sample.
2. Creating a representative sample for each class. 

In each case, we took the class with the least DTW distance to the test sample.

- **Case 1**: Taking DTW distance with respect to each training example and selecting the class of the minimum distance.  
  **Accuracy**: 0.98333

- **Case 2**: Taking test and train examples of a single speaker (Jackson) and then taking DTW distance with respect to each training example.  
  **Accuracy**: 1.0

- **Case 3**: Creating class representatives (taking an average of each training sample from all speakers) and then calculating the distance with each representative for each test example (test sample contains 3 speakers present in train).  
  **Accuracy**:  
  - Nicolas: 0.16  
  - Theo: 0.24  
  - Yweweler: 0.16

### Mismatched Case:
In the mismatched case, the DTW model was trained on MFCC features from one set of speakers and tested on MFCC features from different speakers, simulating real-world scenarios to assess its robustness. Similar to matched cases, two approaches were used:
1. Calculating DTW distance from each train sample.
2. Creating a representative sample for each class.

For the mismatched case, we used George, Jackson, and Lucas for training and Nicolas, Theo, and Yweweler for testing.

- **Case 1**: Taking test and train examples of different speakers (Jackson for training and George for testing) and calculating DTW distance with respect to each training example.  
  **Accuracy**: 0.54

- **Case 2**: Creating class representatives (taking an average of each training sample from 3 speakers George, Jackson, Lucas) and then calculating distance with each representative for each test example (test sample contains a sample from a different speaker than that of the train).  
  **Accuracy**:  
  - Nicolas: 0.1  
  - Theo: 0.1  
  - Yweweler: 0.12

## Isolated Digit Recognition with VGG

### Introduction
This report explores the application of the VGG (Visual Geometry Group) deep learning architecture for isolated digit recognition. VGG, originally designed for image classification tasks, will be adapted and fine-tuned for audio-based recognition, aiming to achieve high accuracy in identifying isolated spoken digits.

### Experiments

#### Audio Representations Used:
For our experiments, we utilized spectrograms as the audio representation. Spectrograms are visual representations of the frequency content of an audio signal over time and serve as an input to the VGG model.

#### Experiment Setup:
We obtained our data from a famous dataset on GitHub known as the free-spoken digit dataset, which contains a diverse dataset of isolated digit utterances from multiple speakers. The speakers include George, Jackson, Lucas, Nicolas, Theo, and Yweweler. The dataset was divided into two subsets: one where the training and testing data belonged to each speaker, and a class-wise case, where the training and testing data came from different classes. We used spectrograms from these `.wav` files.

### Matched Case:
Matched cases include training and testing datasets from the same speakers. We used the entire dataset as both the training and testing sets.  
**Accuracy**: 0.9067

### Mismatched Case:
In the mismatched case, we used spectrograms of 3 speakers George, Jackson, and Lucas for training and Theo, Nicolas, and Yweweler for testing.  
**Accuracy**: 0.2867

## Conclusions

| Model                                    | Accuracy                       |
|------------------------------------------|--------------------------------|
| DTW-Matched Case 1 (Entire dataset)      | 0.9833                         |
| DTW-Matched Case 2 (Single speaker)      | 1                              |
| DTW-Matched Case 3 (Class representative)| 0.16, 0.24 & 0.16              |
| DTW-Mismatched Case 1                    | 0.54                           |
| DTW-Mismatched Case 2 (Class representative) | 0.1, 0.1 & 0.12           |
| VGG-Matched case                         | 0.9067                         |
| VGG-Mismatched case                      | 0.2867                         |

## References
[1] Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*.
