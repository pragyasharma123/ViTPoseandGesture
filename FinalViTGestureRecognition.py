import json
import random
import torchmetrics
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple
from glob import glob
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from transformers import ViTModel, ViTConfig
from PIL import Image, ImageOps
import os
from ipywidgets import interact
from IPython.display import Image as DImage
import cv2

import torch
from torch import nn, Tensor
from torchvision import models
from torchvision.transforms import Compose
from torchvision.transforms import functional as F
from torchvision import transforms as T
import multiprocessing

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision.transforms import ToPILImage
import warnings
warnings.filterwarnings('ignore')


# Path to the saved model weights
model_weights = "/home/pragya/myViT/weights/classification_model.pt"


# THESE ARE THE CLASSNAMES FOR THE 18 DIFFERENT HAND GESTURES
class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace',
   'peace_inverted',
   'rock',
   'stop',
   'stop_inverted',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
   
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureRecognitionModel, self).__init__()
        # Use Hugging Face's transformers library to load the ViT model
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.backbone = vit_backbone
        embedding_size = 768
        self.head = linear_head(embedding_size, num_classes)


    def forward(self, x):
        # Forward pass through the backbone
        outputs  = self.backbone(pixel_values = x) 
        features = outputs.last_hidden_state[:, 0, :]
        classification_output = self.head(features)
       # classification_output = nn.functional.softmax(classification_output, dim=-1)
        return classification_output


class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, path_annotation, path_images, is_train, transform=None, target_image_size=(224, 224)):
        self.is_train = is_train
        self.transform = transform
        self.target_image_size = target_image_size  # Specify the target image size
        self.path_annotation = path_annotation
        self.path_images = path_images
        self.labels = {label: num for (label, num) in
                       zip(class_names, range(len(class_names)))}
        self.annotations = self.__read_annotations(self.path_annotation)

    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple):
        if not os.path.exists(pth):
            print(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        return files

    def __read_annotations(self, path):
        annotations_all = None
        exists_images = []
        for target in class_names:
            path_to_csv = os.path.join(path, f"{target}.json")
            if os.path.exists(path_to_csv):
                json_annotation = json.load(open(
                    os.path.join(path, f"{target}.json")
                ))

                json_annotation = [dict(annotation, **{"name": f"{name}.jpg"}) for name, annotation in
                                   zip(json_annotation, json_annotation.values())]

                annotation = pd.DataFrame(json_annotation)

                annotation["target"] = target
                annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
                exists_images.extend(
                    self.__get_files_from_dir(os.path.join(self.path_images, target), FORMATS))
            else:
                if target != 'no_gesture':
                    print(f"Database for {target} not found")

        annotations_all["exists"] = annotations_all["name"].isin(exists_images)

        annotations_all = annotations_all[annotations_all["exists"]]

        users = annotations_all["user_id"].unique()
        users = sorted(users)
        random.Random(42).shuffle(users)
        train_users = users[:int(len(users) * 0.8)]
        val_users = users[int(len(users) * 0.8):]

        annotations_all = annotations_all.copy()

        if self.is_train:
            annotations_all = annotations_all[annotations_all["user_id"].isin(train_users)]
        else:
            annotations_all = annotations_all[annotations_all["user_id"].isin(val_users)]

        return annotations_all

   
    def __len__(self):
        return self.annotations.shape[0]

    def get_sample(self, index: int):
        row = self.annotations.iloc[[index]].to_dict('records')[0]
        image_pth = os.path.join(self.path_images, row["target"], row["name"])
        image = Image.open(image_pth).convert("RGB")

        # Apply transformations here
        image = self.transform(image)

        # Create a binary label vector of length 19
        label_vector = torch.zeros(19, dtype=torch.float32)
        for label in row["labels"]:
            label_vector[self.labels[label]] = 1.0

        return image, {"image": image, "labels": label_vector}


    def __getitem__(self, index: int):
        image, data = self.get_sample(index)
        return image, torch.argmax(data["labels"], dim=0)


   
transform = Compose([
    Resize((224, 224)),  # Resize to a multiple of 14 for patch embedding compatibility
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

def collate_fn(batch):
    #print(batch)
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return images, labels


def main():

    # Set the multiprocessing start method to 'spawn'
    #multiprocessing.set_start_method('spawn', True)

    # Create an instance of the GestureDataset class
    train_data = GestureDataset(
        path_images='/home/ps332/hagrid/dataset/train/ann_train_images',
        path_annotation='/home/ps332/hagrid/dataset/train/ann_train_val',
        is_train=True,
        transform=transform,
        target_image_size=(224,224)
    )

    test_data = GestureDataset(
        path_images='/home/ps332/hagrid/dataset/test/test_images',
        path_annotation='/home/ps332/hagrid/dataset/test/ann_test',
        is_train=False,
        transform=transform,
        target_image_size=(224,224)
    )

    batch_size = 64
    random_seed = 42
    momentum = 0.9
    num_epoch = 10
    num_classes = len(class_names)
    learning_rate = 0.005
    weight_decay = 5e-4

    torch.manual_seed(random_seed)
    random.seed(random_seed)
   
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=2)

    # Device selection (CUDA GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    # Initialize the Gesture Recognition model
    model = GestureRecognitionModel(num_classes=num_classes)
    # move it to the device (e.g., GPU)
    model.to(device)
    print(model)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0.00001)

    
    print_interval = 10
    best_accuracy = 0.0  # Initialize the best_accuracy variable

    print("training starts!")
    # Training loop
    for epoch in range(num_epoch):

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = [img.to(device) for img in images]  # Move images to device
       
            optimizer.zero_grad()

            # Prepare images for model 
            inputs = torch.stack(images).to(device)  # Convert to a tensor    

            outputs = model(inputs) # Forward pass through model

            labels = labels.squeeze().to(device)
           
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()


            train_loss += loss.item() * len(images)
           
            # Calculate accuracy 
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_accuracy = train_correct / train_total * 100
            print(f"train accuracy:", train_accuracy )

           

            if (batch_idx + 1) % print_interval == 0:
                print(f"Epoch [{epoch + 1}/{num_epoch}] Batch [{batch_idx + 1}/{len(train_loader)}]")

        train_accuracy = train_correct / train_total
        print(f"Epoch [{epoch + 1}/{num_epoch}] Training Loss: {train_loss/train_total:.4f}, Accuracy: {train_accuracy * 100:.4f}")
            
        # Update learning rate with scheduler
        scheduler.step()


    # Save the trained model
        torch.save(model.state_dict(), 'trained_model.pth')

        # Validation loop
        model.eval()  # Set to evaluation mode
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():  # Turn off gradient computation for evaluation
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = [img.to(device) for img in images]  # Move images to device

                # Prepare images for  model (patch embedding and positional encoding)
                inputs = torch.stack(images).to(device)  # Convert to a tensor    

                outputs = model(inputs)  

                labels = labels.squeeze().to(device)
           
                loss = criterion(outputs, labels)
            
                # Update the validation loss
                val_loss += loss.item() * len(images)
            
           
                    # Calculate accuracy for the current batch
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            

                if (batch_idx + 1) % print_interval == 0:
                    print(f"Epoch [{epoch + 1}/{num_epoch}] Batch [{batch_idx + 1}/{len(test_loader)}]")

        val_accuracy = val_correct / val_total
        print(f"Epoch [{epoch + 1}/{num_epoch}] Validation Loss: {val_loss/val_total:.4f}, Accuracy: {val_accuracy * 100:.4f}")

# Save the model if this is the best accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Training complete. Best validation accuracy: {best_accuracy * 100:.4f}")
            

if __name__ == '__main__':
    main()    
