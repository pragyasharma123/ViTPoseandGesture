# ViTPoseandGesture

This repository is an ongoing work for using Google's ViT as a backbone for both Pose Estimation and Gesture Recognition. 

**VitPoseEstimation**:

**Code Components
Imports and Setup**:
The code begins by importing necessary libraries including PyTorch, torchvision, transformers, PIL, and others for image processing and deep learning tasks.
It sets a random seed for reproducibility and selects the appropriate device (GPU or CPU) for training.

**PoseEstimationDataset Class**:
This class inherits from torch.utils.data.Dataset and is responsible for loading and processing the MPII Pose Estimation dataset.
It includes methods for reading data from JSON annotations, performing transformations on images, and normalizing keypoints.

**PoseEstimationModule Class**:
A PyTorch module that defines the pose estimation model.
It utilizes the Vision Transformer (ViT) as a backbone and includes a regression head for predicting keypoint coordinates.
The model predicts normalized keypoint coordinates for a specified number of people and keypoints.

**Data Preparation**:
**To gather annotations file for dataset**: download mpii_data.json
**To gather images**: Visit  **http://human-pose.mpi-inf.mpg.de/#download** and click 'Images'to begin download. Extract images into same folder as the rest of your data. 
**Remember to change json_path and img_dir to your own paths**
The code includes sections for loading and transforming data using the PoseEstimationDataset class.
The dataset is split into training and validation sets, and DataLoader objects are created for each.

**Model Training and Validation**:
Defines a training loop where the model is trained on the pose estimation task.
Uses Mean Squared Error (MSE) loss for the regression of keypoints and an Adam optimizer with a learning rate scheduler.
Includes a validation step within the training loop to monitor model performance on unseen data.
Utilizes tqdm for progress visualization during training and validation.

**Features**:
The implementation is modular, allowing easy adaptation for different datasets and tasks.
Provides a detailed example of using Vision Transformers for a complex task like pose estimation and gesture recognition.
Includes both training and validation steps, along with loss calculations and optimizer adjustments.

**Usage**:
Users can run the provided Python Script to train the model on their dataset.
Adjustments to dataset paths, model parameters, and training settings can be made within the notebook

**ViTGestureRecognition**
This repository contains a comprehensive implementation of a gesture recognition system using Vision Transformers (ViT). The project utilizes state-of-the-art techniques in deep learning and computer vision to accurately classify different hand gestures.

**Components of the Code**
**Libraries and Modules**:
The code extensively uses libraries such as torch, torchvision, transformers, numpy, pandas, matplotlib, and torchmetrics.
Image processing is handled with PIL and OpenCV.
The implementation is done in Python with the support of torch.nn for neural network layers and torch.utils.data for dataset handling.

**Dataset Handling:**
**GestureDataset class:** Handles loading and preprocessing of gesture images and annotations. It's designed to read data from specified directories, apply transformations, and prepare data for the model.
**HAGRID DATASET DOWNLOAD**: Visit **https://github.com/hukenovs/hagrid**. Scroll to 'Downloads' and follow instructions for correct downloading of the dataset. Make sure to change paths to your own in terms of annotations and images. 

The Structure of the dataset is as follows:

├── hagrid_dataset <PATH_TO_DATASET_FOLDER>
│   ├── call
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
├── hagrid_annotations
│   ├── train <PATH_TO_JSON_TRAIN>
│   │   ├── call.json
│   │   ├── ...
│   ├── val <PATH_TO_JSON_VAL>
│   │   ├── call.json
│   │   ├── ...
│   ├── test <PATH_TO_JSON_TEST>
│   │   ├── call.json
│   │   ├── ...


**Image transformations:** The images are resized, converted to tensors, and normalized using standard mean and standard deviation values.
**Dataset division:** The dataset is divided into training and validation sets for effective model training and evaluation.

**Model Architecture:**
**linear_head class:** Defines a linear head for the ViT model.
**GestureRecognitionModel class**: Incorporates the ViT model as the backbone for feature extraction, with a linear head for classification.
The model is tailored to recognize 18 different hand gestures, using pre-trained weights from google/vit-base-patch16-224.

**Training and Evaluation:**
The training loop includes loss computation, backpropagation, and optimizer steps.
A validation loop evaluates model performance on unseen data, computing accuracy and loss.
Model performance metrics include training and validation loss, and accuracy. These are printed out at regular intervals during training.
The best model based on validation accuracy is saved for future use.

**Utilities:**
Multiprocessing for efficient data loading.
Progress bars using tqdm for monitoring training and validation steps.
Warnings are suppressed for cleaner outputs.

**Usage:**
The script can be executed as a main program to train the gesture recognition model.
Users need to specify the paths for the saved model weights, training images, and annotation files.
The model can be trained by running the script, which will automatically handle the training and validation processes.



