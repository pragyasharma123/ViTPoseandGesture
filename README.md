# ViTPoseandGesture

This repository is an ongoing work for using Google's ViT as a backbone for both Pose Estimation and Gesture Recognition. 

**VitPoseEstimation**:

**Code Components
Imports and Setup**
The code begins by importing necessary libraries including PyTorch, torchvision, transformers, PIL, and others for image processing and deep learning tasks.
It sets a random seed for reproducibility and selects the appropriate device (GPU or CPU) for training.

**PoseEstimationDataset Class**
This class inherits from torch.utils.data.Dataset and is responsible for loading and processing the MPII Pose Estimation dataset.
It includes methods for reading data from JSON annotations, performing transformations on images, and normalizing keypoints.

**PoseEstimationModule Class**
A PyTorch module that defines the pose estimation model.
It utilizes the Vision Transformer (ViT) as a backbone and includes a regression head for predicting keypoint coordinates.
The model predicts normalized keypoint coordinates for a specified number of people and keypoints.

**Data Preparation**
**To gather annotations file for dataset**: download mpii_data.json
**To gather images**: Visit  **http://human-pose.mpi-inf.mpg.de/#download** and click 'Images'to begin download. Extract images into same folder as the rest of your data. 
The code includes sections for loading and transforming data using the PoseEstimationDataset class.
The dataset is split into training and validation sets, and DataLoader objects are created for each.

**Model Training and Validation**
Defines a training loop where the model is trained on the pose estimation task.
Uses Mean Squared Error (MSE) loss for the regression of keypoints and an Adam optimizer with a learning rate scheduler.
Includes a validation step within the training loop to monitor model performance on unseen data.
Utilizes tqdm for progress visualization during training and validation.

**Features**
The implementation is modular, allowing easy adaptation for different datasets and tasks.
Provides a detailed example of using Vision Transformers for a complex task like pose estimation and gesture recognition.
Includes both training and validation steps, along with loss calculations and optimizer adjustments.

**Usage**
Users can run the provided Python Script to train the model on their dataset.
Adjustments to dataset paths, model parameters, and training settings can be made within the notebook
