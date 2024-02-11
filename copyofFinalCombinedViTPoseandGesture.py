import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import json
import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import json
import random
from typing import Tuple
import pandas as pd
import os
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

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
    

class PoseEstimationDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.transform = transform or Compose([Resize(target_size), ToTensor()])
        with open(json_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image_filename'])
        image = Image.open(image_path)
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)

        keypoints = []  # Use a list to collect non-zero keypoints
        denormalized_keypoints = []  # Collect all keypoints, including zeros, then filter
        for joint_data in item['ground_truth'].values():
            for joint in joint_data:
                x, y = joint[:2]  # Only take x and y, ignoring visibility
                denormalized_keypoints.append([x, y])
                if not (x == 0 and y == 0):  # Filter out (0.0, 0.0) keypoints for normalized
                    keypoints.append([x / orig_width, y / orig_height])

        # Convert the list of keypoints to a tensor
        keypoints_tensor = torch.tensor(keypoints).float()

        # Convert denormalized keypoints list to a tensor, then filter out (0.0, 0.0) keypoints
        denormalized_keypoints_tensor = torch.tensor(denormalized_keypoints).float()
        valid_indices = denormalized_keypoints_tensor != torch.tensor([0.0, 0.0]).float()
        valid_indices = valid_indices.all(dim=1)  # Ensure both x and y are non-zero
        denormalized_keypoints_tensor = denormalized_keypoints_tensor[valid_indices]

        return image, keypoints_tensor, denormalized_keypoints_tensor, item['image_filename'], orig_width, orig_height
    

max_people = 13
num_keypoints = 16

class linear_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=18):
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
class CombinedPoseAndGestureModel(nn.Module):
    def __init__(self, num_keypoints, max_people=13, num_classes=18):
        super().__init__()
        # Shared ViT Backbone
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.vit_backbone = vit_backbone
        embedding_size = 768
        self.gesture_head = linear_head(384, num_classes)

        # Task-specific adaptation layer for gesture recognition
        self.gesture_adaptation_layer = nn.Sequential(
            nn.Linear(embedding_size, 384),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific adaptation layers for pose estimation
        self.pose_adaptation_layers = nn.Sequential(
            nn.Linear(768, 512),  # Adjust these dimensions as needed
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Pose Estimation Components
        self.pose_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        hidden_size = 256  # Adjusted based on adaptation layer output
        self.pose_combined_processor = nn.Sequential(
            nn.Linear(512 + hidden_size, hidden_size),  # Combine CNN and adapted ViT features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2 * num_keypoints * max_people),
        )


    def forward(self, x):
        # Process with shared ViT backbone
        vit_outputs = self.vit_backbone(x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]
        
        # Adapt features for gesture recognition separately
        gesture_adapted_features = self.gesture_adaptation_layer(vit_features)
        gesture_classification_output = self.gesture_head(gesture_adapted_features)
        
        
        # Adapt features for pose estimation and process with CNN
        pose_features = self.pose_adaptation_layers(vit_features)
        cnn_features = self.pose_feature_extractor(x)
        pose_combined_features = torch.cat((cnn_features, pose_features), dim=1)
        keypoints = self.pose_combined_processor(pose_combined_features)
        keypoints = keypoints.view(-1, max_people, num_keypoints, 2)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1]

        return keypoints, gesture_classification_output


# Parameters for instantiation
num_keypoints = 16  # Assuming 16 keypoints per person
max_people = 13

# Utility functions for Pose Estimation
def filter_keypoints_by_variance(keypoints, variance_threshold=0.01):
    """
    Filter keypoints based on variance across the batch.
    Keypoints with low variance are likely to be less accurate.

    :param keypoints: Predicted keypoints, tensor of shape (batch_size, max_people, num_keypoints, 2).
    :param variance_threshold: Variance threshold for filtering.
    :return: Filtered keypoints tensor.
    """
    # Calculate variance across the batch dimension
    variances = torch.var(keypoints, dim=0)  # Shape: (max_people, num_keypoints, 2)

    # Identify keypoints with variance below the threshold
    low_variance_mask = variances < variance_threshold

    # Filter out low-variance keypoints by setting them to zero
    # Note: This step merely invalidates low-variance keypoints without removing them.
    # You may need to adjust this logic based on how you want to handle filtered keypoints.
    filtered_keypoints = keypoints.clone()
    filtered_keypoints[:, low_variance_mask] = 0  # Set low-variance keypoints to zero

    return filtered_keypoints

    
def calculate_accuracy(valid_predictions, valid_gt, threshold=0.05):
    """
    Calculate the accuracy of valid predictions against the ground truth keypoints.

    :param valid_predictions: Tensor of matched predicted keypoints, shape (N, 2) where N is the number of matched keypoints.
    :param valid_gt: Tensor of ground truth keypoints corresponding to the matched predictions, shape (N, 2).
    :param threshold: Distance threshold to consider a prediction as correct.
    :return: Accuracy as a percentage of correctly predicted keypoints.
    """
    if valid_predictions.numel() == 0 or valid_gt.numel() == 0:
        return 0.0  # Return 0 accuracy if there are no keypoints to compare

    # Calculate the Euclidean distance between each pair of valid predicted and ground truth keypoints
    distances = torch.norm(valid_predictions - valid_gt, dim=1)

    # Determine which predictions are within the threshold distance of the ground truth keypoints
    correct_predictions = distances < threshold

    # Calculate accuracy as the percentage of predictions that are correct
    accuracy = torch.mean(correct_predictions.float()) * 100  # Convert fraction to percentage

    return accuracy.item()


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints_tensors = [item[1] for item in batch]  # List of tensors
    denormalized_keypoints_tensors = [item[2] for item in batch]  # List of tensors
    image_filenames = [item[3] for item in batch]
    orig_widths = torch.tensor([item[4] for item in batch])
    orig_heights = torch.tensor([item[5] for item in batch])

    # Since images can be stacked into a single tensor directly,
    # we leave them as is. For variable-sized tensors like keypoints,
    # we keep them as lists of tensors.

    return images, keypoints_tensors, denormalized_keypoints_tensors, image_filenames, orig_widths, orig_heights

def threshold_filter_keypoints(keypoints, lower_bound=0.05, upper_bound=0.95):
    """
    Filter keypoints based on a simple thresholding mechanism.
    
    :param keypoints: The keypoints predicted by the model, shaped as (batch_size, max_people, num_keypoints, 2).
    :param lower_bound: Lower bound for valid keypoint values.
    :param upper_bound: Upper bound for valid keypoint values.
    :return: Thresholded keypoints tensor.
    """
    # Create a mask for keypoints that fall within the specified bounds
    valid_mask = (keypoints > lower_bound) & (keypoints < upper_bound)
    
    # Apply the mask to both dimensions of the keypoints (x and y)
    valid_keypoints = keypoints * valid_mask.all(dim=-1, keepdim=True)

    return valid_keypoints

def calculate_distances(pred_keypoints, gt_keypoints):
    """
    Calculate distances between predicted keypoints and ground truth keypoints.

    :param pred_keypoints: Predicted keypoints as a tensor of shape (num_predictions, 2).
    :param gt_keypoints: Ground truth keypoints as a tensor of shape (num_gt_keypoints, 2).
    :return: A tensor of distances of shape (num_predictions, num_gt_keypoints).
    """
    num_predictions = pred_keypoints.shape[0]
    num_gt = gt_keypoints.shape[0]
    distances = torch.zeros((num_predictions, num_gt))

    for i in range(num_predictions):
        for j in range(num_gt):
            distances[i, j] = torch.norm(pred_keypoints[i] - gt_keypoints[j])

    return distances

def match_keypoints(distances, threshold=0.05):
    """
    Match predicted keypoints to ground truth keypoints based on minimum distance.

    :param distances: A tensor of distances between predictions and ground truth keypoints.
    :param threshold: Distance threshold for valid matches.
    :return: Indices of predicted keypoints that match ground truth keypoints.
    """
    matched_indices = []

    for i in range(distances.shape[1]):  # Iterate over ground truth keypoints
        min_dist, idx = torch.min(distances[:, i], dim=0)
        if min_dist < threshold:
            matched_indices.append(idx.item())

    return matched_indices

def denormalize_keypoints(keypoints, orig_width, orig_height):
    """
    Denormalize keypoints from [0, 1] range back to original image dimensions.

    :param keypoints: Tensor of keypoints in normalized form, shape (N, 2) where N is the number of keypoints.
    :param orig_width: Original width of the image.
    :param orig_height: Original height of the image.
    :return: Denormalized keypoints tensor.
    """
    denormalized_keypoints = keypoints.clone()
    denormalized_keypoints[:, 0] *= orig_width  # Scale x coordinates
    denormalized_keypoints[:, 1] *= orig_height  # Scale y coordinates
    return denormalized_keypoints


#############################################################################################################################################

# Gesture Recognition

g_transform = Compose([
    Resize((224, 224)),  # Resize to a multiple of 14 for patch embedding compatibility
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

def g_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

#############################################################################################################################################

# Pose estimation

json_path = '/home/ps332/myViT/data/mpii_data.json'
image_dir = '/home/ps332/myViT/data/mpii_data/images/images'

    # Define any transforms you want to apply to your images
p_transforms = Compose([
        Resize((224, 224)),  # Resize the image
        ToTensor(),  # Convert the image to a PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])

    # Instantiate the dataset
dataset = PoseEstimationDataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=p_transforms
    )

def p_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints_tensors = [item[1] for item in batch]  # List of tensors
    denormalized_keypoints_tensors = [item[2] for item in batch]  # List of tensors
    image_filenames = [item[3] for item in batch]
    orig_widths = torch.tensor([item[4] for item in batch])
    orig_heights = torch.tensor([item[5] for item in batch])

    # Since images can be stacked into a single tensor directly,
    # we leave them as is. For variable-sized tensors like keypoints,
    # we keep them as lists of tensors.

    return images, keypoints_tensors, denormalized_keypoints_tensors, image_filenames, orig_widths, orig_heights

def save_checkpoint(epoch, model, g_optimizer, p_optimizer, train_accuracy, total_gesture_loss, total_pose_loss, filename="combined_model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'p_optimizer_state_dict': p_optimizer.state_dict(),
        'train_accuracy': train_accuracy,
        'total_gesture_loss': total_gesture_loss,
        'total_pose_loss': total_pose_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def train_epoch(model, device, gesture_train_loader, pose_train_loader, g_optimizer, p_optimizer, g_criterion, epoch):
    model.train()
    train_total = 0
    train_correct = 0
   

    for (gesture_images, gesture_labels), (pose_images, pose_keypoints, pose_denormalized_keypoints, pose_image_filenames, pose_orig_widths, pose_orig_heights) in zip(gesture_train_loader, pose_train_loader):
        # Gesture task
        gesture_images = [gesture_image.to(device) for gesture_image in gesture_images]  # Move images to device
        g_optimizer.zero_grad()
        inputs = torch.stack(gesture_images).to(device)
        _, gesture_output = model(inputs)
        gesture_labels = gesture_labels.squeeze().to(device)  # Move labels to GPU if available
        g_loss = g_criterion(gesture_output, gesture_labels)
        print("Gesture Loss", g_loss)
        g_loss.backward()  # Perform backpropagation for gesture loss
        g_optimizer.step()  # Update gesture-related parameters

        total_g_loss = g_loss.item() * len(gesture_images)  # For logging purposes only


        # Calculate gesture accuracy
        _, predicted = torch.max(gesture_output, 1)
        train_total += gesture_labels.size(0)
        train_correct += (predicted == gesture_labels).sum().item()
        train_accuracy = train_correct / train_total * 100  
        print(f"Gesture accuracy:" , train_accuracy)

         # snippet to print labels
        for i in range(gesture_labels.size(0)):  # Loop over the batch
                predicted_label_idx = predicted[i].item()
                true_label_idx = gesture_labels[i].item()
                predicted_label_name = class_names[predicted_label_idx]
                true_label_name = class_names[true_label_idx]
                if i == 0:  # Example condition to limit output
                    print(f"Item {i + 1}: Predicted - '{predicted_label_name}', True - '{true_label_name}'")


        # Pose task
        pose_images = pose_images.to(device)
        p_optimizer.zero_grad()
        pose_output, _ = model(pose_images)
        pred_keypoints_flat = pose_output.view(pose_output.size(0), -1, 2)
       

        for i, gt_keypoints in enumerate(pose_keypoints):
            gt_keypoints = gt_keypoints.to(device)
            distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
            matched_indices = match_keypoints(distances, threshold=0.5)  
            valid_predictions = pred_keypoints_flat[i][matched_indices]

            orig_width_single = pose_orig_widths[i].item()
            orig_height_single = pose_orig_heights[i].item()
            denormalized_valid_predictions = denormalize_keypoints(valid_predictions, orig_width_single, orig_height_single)

            original_image_path = os.path.join(image_dir, pose_image_filenames[i])
            if isinstance(pose_denormalized_keypoints[i], list):
                    gt_keypoints_tensor = torch.tensor(pose_denormalized_keypoints[i])
            else:
                    gt_keypoints_tensor = pose_denormalized_keypoints[i]

        p_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints)
        p_loss.backward()  # Perform backpropagation for pose loss
        p_optimizer.step()  # Update pose-related parameters

        print("Pose Loss", p_loss)
        total_p_loss = p_loss.item()  # For logging purposes only

        accuracy = calculate_accuracy(valid_predictions, gt_keypoints)
        print("Pose accuracy", accuracy)

    return train_accuracy, total_g_loss, total_p_loss


def validate_epoch(model, device, gesture_val_loader, pose_val_loader, criterion, epoch):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient computation
        # Gesture Recognition Validation
        gesture_val_loss = 0.0
        gesture_val_correct = 0
        gesture_val_total = 0

        for gesture_images, gesture_labels in gesture_val_loader:
            gesture_images = gesture_images.to(device) 
            _, gesture_output = model(gesture_images)
            gesture_labels  = gesture_labels.squeeze().to(device)
           
            loss = criterion(gesture_output, gesture_labels)
            gesture_val_loss += loss.item()

            _, predicted = torch.max(gesture_output, 1)
            gesture_val_total += gesture_labels.size(0)
            gesture_val_correct += (predicted == gesture_labels).sum().item()

        gesture_val_accuracy = 100 * gesture_val_correct / gesture_val_total
        print(f"Gesture Validation Accuracy: {gesture_val_accuracy:.2f}%")

        # Pose Estimation Validation
        pose_val_loss = 0.0
        pose_val_accuracy_list = []

        for pose_images, pose_keypoints, pose_denormalized_keypoints, pose_image_filenames, pose_orig_widths, pose_orig_heights in pose_val_loader:
            pose_images = pose_images.to(device)
            pose_output, _ = model(pose_images)
            (f"forward pass done!")
            pred_keypoints_flat = pose_output.view(pose_output.size(0), -1, 2)

            # Compute loss and accuracy for each image/keypoints pair in the batch
            for i, gt_keypoints in enumerate(pose_keypoints):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)  
                valid_predictions = pred_keypoints_flat[i][matched_indices]

                # Example for one image in the batch
                orig_width_single = pose_orig_widths[i].item()  # Assuming i is the index of the current image in the batch
                orig_height_single = pose_orig_heights[i].item()
                # Denormalize valid predictions
                denormalized_valid_predictions = denormalize_keypoints(valid_predictions, orig_width_single, orig_height_single)

                original_image_path = os.path.join(image_dir, pose_image_filenames[i])
                if isinstance(pose_denormalized_keypoints[i], list):
                    gt_keypoints_tensor = torch.tensor(pose_denormalized_keypoints[i])
                else:
                    gt_keypoints_tensor = pose_denormalized_keypoints[i]
                
          

        pose_val_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints)
        pose_val_loss += pose_val_loss.item()
        print("Pose loss" , pose_val_loss)

        total_loss = gesture_val_loss + pose_val_loss


        # Average pose validation accuracy
        pose_val_accuracy = sum(pose_val_accuracy_list) / len(pose_val_accuracy_list) if pose_val_accuracy_list else 0
        print(f"Pose Validation Accuracy: {pose_val_accuracy:.2f}%")

num_classes = len(class_names)
def main():
    
    # Device selection (CUDA GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.cuda.empty_cache()

   # Instantiate the combined model
    model = CombinedPoseAndGestureModel(num_keypoints, max_people, num_classes = num_classes).to(device)
    
      # Instaniate the Gesture Dataset 
    train_data = GestureDataset(
        path_images='/home/ps332/hagrid/dataset/train/ann_train_images',
        path_annotation='/home/ps332/hagrid/dataset/train/ann_train_val',
        is_train=True,
        transform=g_transform,
        target_image_size=(224,224)
    )

    test_data = GestureDataset(
        path_images='/home/ps332/hagrid/dataset/test/test_images',
        path_annotation='/home/ps332/hagrid/dataset/test/ann_test',
        is_train=False,
        transform=g_transform,
        target_image_size=(224,224)
    )

    # Instantiate the Pose dataset
    dataset = PoseEstimationDataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=p_transforms
    )
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    # Split the Pose dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

   
   # Gesture Recognition components
    gesture_train_loader = DataLoader(train_data, batch_size=64, collate_fn=g_collate_fn, shuffle=True, num_workers=2)
    gesture_val_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=g_collate_fn, num_workers=2)
    g_optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    g_criterion = nn.CrossEntropyLoss()
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=10, eta_min=0.00001)


    # Pose estimation components
    pose_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=p_collate_fn)
    pose_val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=p_collate_fn)
    p_optimizer = optim.Adam(model.parameters(), lr=5e-4)
    p_scheduler = StepLR(p_optimizer, step_size=10, gamma=0.1)

    num_epoch = 10
    for epoch in range(num_epoch):
        # Calling train_epoch and accessing returned values
        train_accuracy, total_g_loss, total_p_loss = train_epoch(model, device, gesture_train_loader, pose_train_loader, g_optimizer, p_optimizer, g_criterion, epoch)
        save_checkpoint(epoch, model, g_optimizer, p_optimizer, train_accuracy, total_g_loss, total_p_loss)
        #validate_epoch(model, device, gesture_val_loader, pose_val_loader, g_criterion, epoch)
        g_scheduler.step()
        p_scheduler.step()
    
    
if __name__ == '__main__':
    main()    
