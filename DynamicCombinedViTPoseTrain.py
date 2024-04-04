import random
from torch.utils.data import DataLoader, Subset
from transformers import ViTModel, ViTConfig
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch
from transformers import ViTModel, ViTConfig
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import GestureDataset
import warnings
warnings.filterwarnings('ignore')

# Device selection (CUDA GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

num_keypoints = 16
num_classes = len(class_names)
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)


class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        if activation_func == "ReLU":
            self.activation = nn.ReLU()
        elif activation_func == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation_func == "ELU":
            self.activation = nn.ELU()

    def forward(self, x):
        return self.dropout(self.activation(self.bn(self.linear(x))))

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, layer_sizes, dropout_rates, activations):
        super(GestureRecognitionHead, self).__init__()
        layers = []
        input_size = embedding_size

        for i, (size, dropout_rate, activation) in enumerate(zip(layer_sizes, dropout_rates, activations)):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)


class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13, pose_layer_sizes=[512, 256], pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"]):
        super(DynamicPoseEstimationHead, self).__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        
        for layer_size, dropout_rate, activation in zip(pose_layer_sizes, pose_dropout_rates, pose_activations):
            self.layers.append(DynamicLinearBlock(input_size, layer_size, dropout_rate, activation))
            input_size = layer_size
            
        self.output_layer = nn.Linear(input_size, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)  # Assuming the last dimension is 2 for (x, y) coordinates
        return x



class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, gesture_layer_sizes=[1024, 512], gesture_dropout_rates=[0.5, 0.4], gesture_activations=["ReLU", "LeakyReLU"], pose_layer_sizes=[512, 256], pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"]):
        super(CombinedModel, self).__init__()
        self.max_people = max_people

        # Gesture Recognition Head (frozen during the training of Pose Estimation Head)
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates, activations=gesture_activations)
        for param in self.gesture_head.parameters():
            param.requires_grad = False  # Freeze the gesture recognition head

        # Vision Transformer Backbone
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # CNN Feature Processor
        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        hidden_size = self.backbone.config.hidden_size

        # Dynamic Pose Estimation Head
        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=512 + hidden_size, num_keypoints=num_keypoints, max_people=max_people, pose_layer_sizes=pose_layer_sizes, pose_dropout_rates=pose_dropout_rates, pose_activations=pose_activations)

    def forward(self, x):
        with torch.no_grad():  # Keeping gesture recognition head frozen
            vit_outputs = self.backbone(pixel_values=x)
            vit_features = vit_outputs.last_hidden_state[:, 0, :]
            gesture_output = self.gesture_head(vit_features)

        # CNN features processing
        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)
        combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)

        # Pose estimation
        keypoints = self.pose_estimation_head(combined_features)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range
        
        return keypoints, gesture_output


import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


# Pose Dataset
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


# Pose Utility Functions:
import torch

# Device selection (CUDA GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# pose estimation utility functions
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

def calculate_valid_accuracy(pred_keypoints, gt_keypoints, threshold=0.05):
    """
    Calculate accuracy based on the distance between predicted and ground truth keypoints,
    considering only those keypoints that are matched within a specified threshold.
    """
    total_correct = 0
    total_valid = 0

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        # Assuming pred and gt are already on the correct device and properly scaled
        distances = calculate_distances(pred, gt)
        matched = match_keypoints(distances, threshold)
        
        total_correct += len(matched)
        total_valid += gt.size(0)  # Assuming gt is a 2D tensor of shape [N, 2]

    if total_valid > 0:
        accuracy = (total_correct / total_valid) * 100
    else:
        accuracy = 0.0

    return accuracy

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
    pred_keypoints = pred_keypoints.to(device)
    gt_keypoints = gt_keypoints.to(device)
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

def pad_tensors_to_match(a, b):
    """
    Pad the shorter tensor among 'a' and 'b' with zeros to match the length of the longer tensor.
    Returns padded tensors and a mask indicating the original elements.
    
    Args:
    a (Tensor): First tensor.
    b (Tensor): Second tensor.
    
    Returns:
    Tensor, Tensor, Tensor: Padded version of 'a', padded version of 'b', and a mask.
    """
    max_len = max(a.size(0), b.size(0))
    
    # Create masks for original keypoints (1 for real, 0 for padded)
    mask_a = torch.ones(a.size(0), dtype=torch.float32, device=a.device)
    mask_b = torch.ones(b.size(0), dtype=torch.float32, device=b.device)
    
    # Pad tensors to match the maximum length
    padded_a = torch.cat([a, torch.zeros(max_len - a.size(0), *a.shape[1:], device=a.device)], dim=0)
    padded_b = torch.cat([b, torch.zeros(max_len - b.size(0), *b.shape[1:], device=b.device)], dim=0)
    
    # Pad masks to match the maximum length
    padded_mask_a = torch.cat([mask_a, torch.zeros(max_len - mask_a.size(0), device=a.device)], dim=0)
    padded_mask_b = torch.cat([mask_b, torch.zeros(max_len - mask_b.size(0), device=b.device)], dim=0)
    
    # Combine masks (logical AND) since we want to consider keypoints that are present in both tensors
    combined_mask = padded_mask_a * padded_mask_b
    
    return padded_a, padded_b, combined_mask

import torch.nn.functional as F

def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss between 'pred' and 'target', applying 'mask' to ignore padded values.
    
    Args:
    pred (Tensor): Predicted keypoints.
    target (Tensor): Ground truth keypoints.
    mask (Tensor): Mask tensor indicating valid keypoints.
    
    Returns:
    Tensor: Masked MSE loss.
    """
    # Ensure the mask is boolean for advanced indexing
    mask = mask.bool()
    
    # Flatten the tensors and mask to simplify indexing
    pred_flat = pred.view(-1, pred.size(-1))
    target_flat = target.view(-1, target.size(-1))
    mask_flat = mask.view(-1)
    
    # Apply mask
    valid_pred = pred_flat[mask_flat]
    valid_target = target_flat[mask_flat]
    
    # Compute MSE loss on valid keypoints only
    loss = F.mse_loss(valid_pred, valid_target)
    
    return loss

    # Define any transforms you want to apply to your images
p_transforms = Compose([
        Resize((224, 224)),  # Resize the image
        ToTensor(),  # Convert the image to a PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])


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



def main():

    json_path = '/home/ps332/myViT/data/mpii_data.json'
    image_dir = '/home/ps332/myViT/data/mpii_data/images/images'

    # Instantiate the dataset
    dataset = PoseEstimationDataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=p_transforms
    )
    # Calculate subset size (20% of the entire dataset)
    total_size = len(dataset)
    subset_size = int(0.2 * total_size)

    # Calculate training and validation sizes from the subset
    train_size = int(0.8 * subset_size)  # 80% of the subset for training
    val_size = subset_size - train_size  # Remaining 20% of the subset for validation

    # Generate indices for the entire dataset
    indices = torch.randperm(total_size).tolist()

    # Select indices for the subset (first 20% of the shuffled indices)
    subset_indices = indices[:subset_size]

    # Now split the subset indices into training and validation indices
    train_indices = subset_indices[:train_size]
    val_indices = subset_indices[train_size:]

    # Create subsets for training and validation using Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create DataLoader for training and validation datasets
    p_train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=p_collate_fn, shuffle=True)
    p_val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=p_collate_fn, shuffle=False)
    

    random_seed = 42
    num_epoch = 10
    num_classes = len(class_names)

    # Device selection (CUDA GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(random_seed)
    random.seed(random_seed)

    

    # Initialize the Gesture Recognition model
    model = CombinedModel(num_classes=num_classes, num_keypoints=16)
    # move it to the device (e.g., GPU)
    model.to(device)
    print(model)

    # Freeze pose estimation components
    for param in model.gesture_head.parameters():
        param.requires_grad = False

    # shared optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    p_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

   
    

    for epoch in range(num_epoch):
        
        model.train()  # Ensure the model is in training mode
        total_train_loss = 0.0
        total_accuracy = 0.0

        for batch_idx, (images, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in enumerate(p_train_dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            output, _ = model(images)
            pred_keypoints_flat = output.view(output.size(0), -1, 2)  

            batch_loss, batch_accuracy = 0.0, 0.0 
            # Iterate over all keypoints in the batch
            for i, gt_keypoints in enumerate(keypoints_tensor_list):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)  
                valid_predictions = pred_keypoints_flat[i][matched_indices]
                
                #print(f"valid_predictions shape: {valid_predictions.shape}")
                #print(f"gt_keypoints shape: {gt_keypoints.shape}")
                #print(f"matched_indices: {matched_indices}")
               # Now you can use .max() and .min() on the NumPy array
                #print(f"Max index: {matched_indices_np.max()}, Min index: {matched_indices_np.min()}")


                # Convert matched_indices to a tensor for easy manipulation
                matched_indices_tensor = torch.tensor(matched_indices, device=device)

                # Filter out out-of-bounds indices
                valid_indices = matched_indices_tensor[(matched_indices_tensor >= 0) & (matched_indices_tensor < gt_keypoints.size(0))]

                # Now use valid_indices to index into gt_keypoints safely
                if len(valid_indices) > 0:  # Ensure there are any valid indices to index with
                    valid_predictions = valid_predictions[valid_indices]  # Assuming valid_predictions needs to be filtered similarly
                    gt_keypoints_subset = gt_keypoints[valid_indices]
                    keypoint_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints_subset)
                    # Calculate accuracy for the valid predictions
                    batch_accuracy += calculate_accuracy(valid_predictions, gt_keypoints_subset)
                else:
                    keypoint_loss = torch.tensor(0.0, device=device)  # Or handle this case as appropriate
                
                batch_loss += keypoint_loss  # Accumulate the loss

               

            batch_loss.backward()  # Perform a single backpropagation for the accumulated loss
            optimizer.step()

            total_train_loss += batch_loss.item()
            total_accuracy += batch_accuracy / len(keypoints_tensor_list)  # Average accuracy across keypoints in the batch
            print(f"Batch {batch_idx + 1}, Loss: {batch_loss.item()}, Accuracy: {batch_accuracy / len(keypoints_tensor_list):.2f}%")
            

        avg_train_loss = total_train_loss / len(p_train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

       # Save the trained model
        torch.save(model.state_dict(), 'final_vit_pose_trained_combined_model.pth')
    
  

    # Training loop for both tasks 
    '''
    for epoch in range(num_epoch):
        train_loss, train_correct, train_total = 0.0, 0, 0
        for (images, labels), (image, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in zip(g_train_loader, p_train_dataloader):
            # gesture task
            images = [img.to(device) for img in images]  # Move images to device
            optimizer.zero_grad()
            # Prepare images for model 
            inputs = torch.stack(images).to(device)  # Convert to a tensor    
            _, outputs = model(inputs) # Forward pass through model
            labels = labels.squeeze().to(device)
            g_loss = g_criterion(outputs, labels)
            print(f"gesture loss:" , g_loss)
            g_loss.backward()
            train_loss += g_loss.item() * len(images)
           
            # Calculate accuracy 
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_accuracy = train_correct / train_total
            print(f"Gesture accuracy:" , train_accuracy * 100)

            # snippet to print labels
            for i in range(labels.size(0)):  # Loop over the batch
                predicted_label_idx = predicted[i].item()
                true_label_idx = labels[i].item()
                predicted_label_name = class_names[predicted_label_idx]
                true_label_name = class_names[true_label_idx]
                if i == 0:  # Example condition to limit output
                    print(f"Item {i + 1}: Predicted - '{predicted_label_name}', True - '{true_label_name}'")

            # pose task
            image = image.to(device)
            output, _ = model(image)
            pred_keypoints_flat = output.view(output.size(0), -1, 2)  

            for i, gt_keypoints in enumerate(keypoints_tensor_list):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)  
                valid_predictions = pred_keypoints_flat[i][matched_indices]


            p_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints)
            p_loss +=p_loss.item()
            print("pose loss" , p_loss)
            p_loss.backward()
            p_accuracy = calculate_accuracy(valid_predictions, gt_keypoints)
            print("Pose Accuracy", p_accuracy)

            optimizer.step()

        g_train_accuracy = train_correct / train_total
        print(f"Epoch [{epoch + 1}/{num_epoch}] Training Loss: {train_loss/train_total:.4f}, Gesture Accuracy: {g_train_accuracy * 100:.4f}")

        p_accuracy = calculate_accuracy(valid_predictions, gt_keypoints)
        print("Pose Accuracy", p_accuracy)
            
        # Update learning rate with scheduler   
        g_scheduler.step()
        p_scheduler.step()


    # Save the trained model
        torch.save(model.state_dict(), 'final_vit_gesture_trained_model.pth')

#############################################################################################################################################################################
# Validation loop
    model_weights_path = "/home/ps332/myViT/finalizing_vit/final_vit_pose_trained_combined_model.pth"
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()  # Set to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_val_loss = 0.0
    total_val_accuracy = 0.0

    with torch.no_grad():  # No need to track gradients for validation
        for batch_idx, (images, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in enumerate(p_val_dataloader):
            images = images.to(device)
            output, _ = model(images)
            pred_keypoints_flat = output.view(output.size(0), -1, 2)

            batch_loss, batch_accuracy = 0.0, 0.0
            for i, gt_keypoints in enumerate(keypoints_tensor_list):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)
                if not matched_indices:  # Skip if matched_indices is empty
                    continue
                valid_predictions = pred_keypoints_flat[i][matched_indices]

                matched_indices_tensor = torch.tensor(matched_indices, device=device)
                valid_indices = matched_indices_tensor[(matched_indices_tensor >= 0) & (matched_indices_tensor < gt_keypoints.size(0))]


                # Only proceed if there are valid indices
                if valid_indices.numel() > 0:
                    valid_predictions_filtered = valid_predictions[valid_indices]
                    gt_keypoints_subset = gt_keypoints[valid_indices]
                    keypoint_loss = torch.nn.functional.mse_loss(valid_predictions_filtered, gt_keypoints_subset)
                    print(f"Keypoint loss: {keypoint_loss}")
                    batch_accuracy += PoseUtility.calculate_accuracy(valid_predictions_filtered, gt_keypoints_subset)
                else:
                    keypoint_loss = torch.tensor(0.0, device=device)

                batch_loss += keypoint_loss.item()
                print(f"Batch {batch_idx + 1}, Loss: {batch_loss}, Accuracy: {batch_accuracy / len(keypoints_tensor_list):.2f}%")

            total_val_loss += batch_loss
            total_val_accuracy += batch_accuracy / len(keypoints_tensor_list)  # Average accuracy across keypoints in the batch

    avg_val_loss = total_val_loss / len(p_val_dataloader)
    avg_val_accuracy = total_val_accuracy / len(p_val_dataloader)
    print(f"Validation - Epoch {epoch + 1}, Average Loss: {avg_val_loss:.4f}, Average Accuracy: {avg_val_accuracy:.2f}%")

    torch.save(model.state_dict(), 'final_vit_pose_validated_combined_model.pth')

    '''

if __name__ == '__main__':
    main()    
