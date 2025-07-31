 Malignant Breast Cancer Subtype Classification using Deep Learning

## Project Overview

This project implements a deep learning model to classify four different subtypes of malignant breast cancer using histology images from the BreaKHis dataset. The goal is to create an accurate classifier that can distinguish between Ductal Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), and Phyllodes Carcinoma (PC).

The model leverages a multi-branch feature fusion architecture built upon a pre-trained DenseNet121 to achieve high accuracy. This approach extracts features from multiple depths of the network, combines them, and uses the fused representation for final classification.

## Dataset

The BreaKHis dataset is used for this project, focusing exclusively on malignant tumor images. The data is organized into four sub-categories, representing the different malignant subtypes.

  - *Dataset Source:* BreaKHis\_v1
  - *Total Malignant Images:* 5,429
  - *Subtype Distribution:*
      - *DC (Ductal Carcinoma):* 3,451 images
      - *LC (Lobular Carcinoma):* 626 images
      - *MC (Mucinous Carcinoma):* 792 images
      - *PC (Phyllodes Carcinoma):* 560 images

The notebook assumes the dataset is located in a Google Drive folder mounted at /content/drive/MyDrive/BreakHis/.

## Methodology

### 1\. Data Preprocessing and Augmentation

To prepare the data for training and to increase the dataset's diversity, the following steps were taken:

  - *Image Loading:* Images were loaded using OpenCV and converted to RGB format.
  - *Resizing:* All images were resized to a uniform 128x128 pixels.
  - *Normalization:* Pixel values were scaled to a range of [0, 1].
  - *Data Augmentation:* To prevent overfitting and create a more robust model, each image was augmented to generate two additional versions:
      - 90-degree rotation followed by a horizontal flip.
      - 180-degree rotation followed by a horizontal flip.
        This process tripled the total number of training images.

### 2\. Model Architecture

A sophisticated multi-branch feature fusion model was designed using TensorFlow/Keras.

  - *Base Model:* *DenseNet121*, pre-trained on ImageNet, was used as the feature extraction backbone (with the top classification layer removed).
  - *Feature Fusion:* Instead of using only the final feature map, this model extracts features from three intermediate layers of DenseNet121 (conv3_block12_concat, conv4_block24_concat, conv5_block16_concat).
  - *Branch Processing:* Each feature map is fed into a separate branch consisting of:
    1.  Global Average Pooling
    2.  L2 Normalization
    3.  A Dense layer (64 units, ReLU activation)
    4.  Batch Normalization
  - *Concatenation:* The outputs from all three branches are concatenated to create a single, rich feature vector.
  - *Classifier Head:* The fused vector is passed through a final set of Dense, Batch Normalization, and Dropout layers before making a prediction with a 4-class Softmax output layer.

  
(This is a conceptual representation of a multi-branch network. The actual layers are as described above.)

### 3\. Training

  - *Data Split:* The augmented dataset was split into training (60%), validation (20%), and testing (20%) sets.
  - *Optimizer:* Adam with a learning rate of 0.0001.
  - *Loss Function:* categorical_crossentropy.
  - *Callbacks:*
      - EarlyStopping: To halt training when validation loss stops improving (patience=5).
      - ReduceLROnPlateau: To reduce the learning rate if validation loss plateaus (patience=3).
  - *Environment:* The model was trained on a GPU for 34 epochs, taking approximately 536 seconds.

## Results

The model achieved excellent performance on the unseen test set.

  - *Test Accuracy:* *92.82%*

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| *DC* | 0.94 | 0.96 | 0.95 | 2053 |
| *LC* | 0.79 | 0.74 | 0.76 | 389 |
| *MC* | 0.96 | 0.97 | 0.96 | 458 |
| *PC* | 0.98 | 0.91 | 0.94 | 358 |
| *Accuracy* | | | *0.93* | *3258* |
| *Macro Avg* | *0.92* | *0.89* | *0.90* | *3258* |
| *Weighted Avg| **0.93* | *0.93* | *0.93* | *3258* |

### Confusion Matrix

The confusion matrix below shows that the model performs very well, especially for the DC and MC classes. The most confusion occurs with the less-represented LC class.

### Training & Validation Accuracy

## How to Run

1.  *Environment:* This project is designed to run in a Google Colab environment with a GPU accelerator.
2.  *Dependencies:* Install the required libraries.
    bash
    pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn
    
3.  *Dataset:*
      - Download the BreaKHis\_v1 dataset.
      - Upload the malignant folder to your Google Drive, following this path: My Drive/BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant.
4.  *Execution:*
      - Open the malignant.ipynb notebook in Google Colab.
      - Mount your Google Drive when prompted.
      - Run the cells sequentially. The notebook will preprocess the data, build the model, train it, and display the evaluation results.
5.  *Saved Model:* The trained model is saved as malignant_model.h5 and malignant_model.keras at the end of the notebook execution.

## Files in this Repository

  - malignant.ipynb: The Jupyter Notebook containing the complete code for data processing, model training, and evaluation.
    
