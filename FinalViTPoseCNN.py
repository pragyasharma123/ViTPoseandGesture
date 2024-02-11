import json
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torch.utils.data import Dataset
from transformers import ViTModel, ViTConfig
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR


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
    
num_keypoints = 16  # Assuming 16 keypoints per person
class PoseEstimationModule(nn.Module):
    def __init__(self, num_keypoints, max_people=13):
        super().__init__()
        self.max_people = max_people
        # Simplified CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Load the pre-trained ViT model
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.vit_backbone = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)
        hidden_size = self.vit_backbone.config.hidden_size

        # Adjusted layers for processing CNN features before combination
        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),  # Adjusted based on new CNN output size
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Assuming the combination of CNN and ViT features before the regression head
        self.combined_feature_processor = nn.Sequential(
            nn.Linear(512 + hidden_size, hidden_size // 2),  # Combine CNN (512) and ViT (hidden_size) features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 2 * num_keypoints * max_people),
        )

    def forward(self, x):
        # Process with CNN
        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)
        
        # Process the same image with ViT
        outputs = self.vit_backbone(x)
        vit_features = outputs.last_hidden_state[:, 0, :]
        
        # Combine CNN and ViT features
        combined_features = torch.cat((processed_cnn_features, vit_features), dim=1)
        
        # Final prediction
        keypoints = self.combined_feature_processor(combined_features)
        keypoints = keypoints.view(-1, self.max_people, num_keypoints, 2)
        keypoints = torch.sigmoid(keypoints)  # Ensure keypoints are in [0, 1]
        
        return keypoints

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

def plot_keypoints(original_image_path, gt_keypoints_denormalized, pred_keypoints_denormalized, epoch, batch_idx, image_idx, save_dir):
    """
    Plots and saves ground truth and predicted keypoints on the original image.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"epoch_{epoch}_batch_{batch_idx}_image_{image_idx}.png"
    filepath = os.path.join(save_dir, filename)

    fig, ax = plt.subplots()
    original_image = Image.open(original_image_path)
    ax.imshow(original_image)

    # Convert lists to tensors if they are not already tensors
    if isinstance(gt_keypoints_denormalized, list):
        gt_keypoints_denormalized = torch.tensor(gt_keypoints_denormalized)
    if isinstance(pred_keypoints_denormalized, list):
        pred_keypoints_denormalized = torch.tensor(pred_keypoints_denormalized)

    # Plot ground truth keypoints
    if gt_keypoints_denormalized is not None and gt_keypoints_denormalized.numel() > 0:
        ax.scatter(gt_keypoints_denormalized[:, 0], gt_keypoints_denormalized[:, 1], c='blue', label='Ground Truth', s=20)

    # Check and adjust dimensions for pred_keypoints_denormalized if necessary
    if pred_keypoints_denormalized.dim() == 1:
        pred_keypoints_denormalized = pred_keypoints_denormalized.unsqueeze(0)  # Adjust to [1, 2] if it's a single point

    # Plot predicted keypoints
    if pred_keypoints_denormalized.numel() > 0:
        ax.scatter(pred_keypoints_denormalized[:, 0].cpu().detach().numpy(), pred_keypoints_denormalized[:, 1].cpu().detach().numpy(), c='red', label='Predicted', s=20)


    ax.set_title(f"Epoch {epoch}, Batch {batch_idx}, Image {image_idx}")
    ax.legend()
    ax.axis('off')
    plt.savefig(filepath)
    plt.close()

if __name__ == '__main__':

    json_path = '/home/ps332/myViT/data/mpii_data.json'
    image_dir = '/home/ps332/myViT/data/mpii_data/images/images'

    # Define any transforms you want to apply to your images
    transforms = Compose([
        Resize((224, 224)),  # Resize the image
        ToTensor(),  # Convert the image to a PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])

    # Instantiate the dataset
    dataset = PoseEstimationDataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=transforms
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

   # Device selection (CUDA GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # train
    model = PoseEstimationModule(num_keypoints).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    train_loss = 0.0
    num_epochs = 5

    output_data = {}
    save_dir = "/home/ps332/myViT/plots"
    for epoch in range(num_epochs):
        for batch_idx, (image, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in enumerate(train_dataloader):
            image = image.to(device)
            optimizer.zero_grad()
            output = model(image)
            pred_keypoints_flat = output.view(output.size(0), -1, 2)  

            for i, gt_keypoints in enumerate(keypoints_tensor_list):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)  
                valid_predictions = pred_keypoints_flat[i][matched_indices]

                # Example for one image in the batch
                orig_width_single = orig_width[i].item()  # Assuming i is the index of the current image in the batch
                orig_height_single = orig_height[i].item()
                # Denormalize valid predictions
                denormalized_valid_predictions = denormalize_keypoints(valid_predictions, orig_width_single, orig_height_single)

                original_image_path = os.path.join(image_dir, image_filenames[i])
                if isinstance(denormalized_keypoints_list[i], list):
                    gt_keypoints_tensor = torch.tensor(denormalized_keypoints_list[i])
                else:
                    gt_keypoints_tensor = denormalized_keypoints_list[i]
                
                print("Denormalized ground truth keypoints", denormalized_keypoints_list[i])
                # Now denormalized_valid_predictions contains the keypoints in the original image dimensions
                print("Denormalized valid predictions", denormalized_valid_predictions)

                # Plotting images for every batch in epoch
                #plot_keypoints(original_image_path, denormalized_keypoints_list[i], denormalized_valid_predictions, epoch, batch_idx, i, save_dir)

            train_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints)
            train_loss += train_loss.item()
            print("loss" , train_loss)
            train_loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(valid_predictions, gt_keypoints)
            print("accuracy", accuracy)
        
        scheduler.step()    
        # After the training loop and before the validation loop
        train_loss_avg = train_loss / len(train_dataloader)
        print(f"Average Training Loss: {train_loss_avg:.4f}")

        # Optionally, print the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, (image, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in enumerate(val_dataloader):
            image = image.to(device)
            optimizer.zero_grad()
            output = model(image)
            pred_keypoints_flat = output.view(output.size(0), -1, 2)  
            
            for i, gt_keypoints in enumerate(keypoints_tensor_list):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)  
                valid_predictions = pred_keypoints_flat[i][matched_indices]

                # Example for one image in the batch
                orig_width_single = orig_width[i].item()  # Assuming i is the index of the current image in the batch
                orig_height_single = orig_height[i].item()
                # Denormalize valid predictions
                denormalized_valid_predictions = denormalize_keypoints(valid_predictions, orig_width_single, orig_height_single)

                original_image_path = os.path.join(image_dir, image_filenames[i])
                if isinstance(denormalized_keypoints_list[i], list):
                    gt_keypoints_tensor = torch.tensor(denormalized_keypoints_list[i])
                else:
                    gt_keypoints_tensor = denormalized_keypoints_list[i]
                
                print("Denormalized ground truth keypoints", denormalized_keypoints_list[i])
                # Now denormalized_valid_predictions contains the keypoints in the original image dimensions
                print("Denormalized valid predictions", denormalized_valid_predictions)

            loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints)  
            val_loss += loss.item()
            print("val loss" , val_loss)
            
            # Compute accuracy if applicable
            val_accuracy = calculate_accuracy(valid_predictions, gt_keypoints)  
            print(f"Accuracy" , val_accuracy)

    # Average loss and accuracy for the epoch
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    val_accuracy /= len(val_dataloader)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save checkpoint
    checkpoint_path = f"/home/ps332/myViT/finalViTPoseCNNcheckpoint_epoch_{epoch+1}.pth"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
                

