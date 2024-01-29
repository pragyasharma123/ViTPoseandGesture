
from tqdm import tqdm
import json
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torch.utils.data import Dataset
from transformers import ViTModel, ViTConfig
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

def plot_keypoints(image, gt_keypoints, pred_keypoints, epoch, idx, phase):
    """
    Plots and saves an image with ground truth and predicted keypoints.

    :param image: The input image as a tensor.
    :param gt_keypoints: Ground truth keypoints.
    :param pred_keypoints: Predicted keypoints.
    :param epoch: Current epoch number.
    :param idx: Index of the batch.
    :param phase: 'train' or 'val' indicating the phase.
    """
    # Convert the image tensor to PIL Image for visualization
    image = image.cpu().squeeze().permute(1, 2, 0)
    image = (image * torch.tensor([0.229, 0.224, 0.225])) + torch.tensor([0.485, 0.456, 0.406])
    image = image.numpy()
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    
    # Plot ground truth keypoints
    gt_keypoints = gt_keypoints.cpu().numpy()
    plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='blue', label='Ground Truth', s=10)
    
    # Plot predicted keypoints
    pred_keypoints = pred_keypoints.cpu().numpy()
    plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c='red', label='Predicted', s=10)

    plt.legend()
    plt.title(f'Epoch {epoch}, Batch {idx}, {phase.capitalize()}')
    
    # Save the figure
    os.makedirs(f'plots/{phase}', exist_ok=True)
    plt.savefig(f'/home/ps332/myViT/plots/{phase}/epoch_{epoch}_batch_{idx}.png')
    plt.close()

def custom_sigmoid(x):
    return torch.sigmoid(x) * 1.4 - 0.1

class PoseEstimationDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size

        with open(json_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        max_people = 13
        num_keypoints = 16  # Assuming 16 keypoints per person

        image_path = os.path.join(self.image_dir, item['image_filename'])
        image = Image.open(image_path)
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)

        keypoints_tensor = torch.zeros((max_people, num_keypoints, 2))
        
        # Normalize the keypoints
        for i, (joint_name, joint_data) in enumerate(item['ground_truth'].items()):
            for j, joint in enumerate(joint_data):
                if j >= max_people:
                    break  # Skip extra people
                x, y = joint[:2]  # Only take x and y, ignoring visibility
                keypoints_tensor[j, i, 0] = x / orig_width
                keypoints_tensor[j, i, 1] = y / orig_height

        # Denormalize the keypoints
        denormalized_keypoints = keypoints_tensor.clone()
        for person in range(max_people):
            for kpt in range(num_keypoints):
                denormalized_keypoints[person, kpt, 0] *= orig_width
                denormalized_keypoints[person, kpt, 1] *= orig_height

         
        return image, keypoints_tensor, denormalized_keypoints, item['image_filename'], orig_width, orig_height    


num_keypoints = 16  # Assuming 16 keypoints per person
class PoseEstimationModule(nn.Module):
    def __init__(self, num_keypoints, max_people=13):
        super().__init__()
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)
        hidden_size = self.backbone.config.hidden_size

        self.max_people = max_people

        # Head for keypoint coordinates
        self.keypoint_regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 2 * num_keypoints * max_people),
            nn.BatchNorm1d(2 * num_keypoints * max_people)
        )

        
    def forward(self, x):
        outputs = self.backbone(x)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.keypoint_regression_head(x)
        x = x.view(x.size(0), self.max_people, num_keypoints, 2)  # Reshape to [batch_size, max_people, num_keypoints, 2]
        #x = torch.sigmoid(x)  # Apply sigmoid to constrain values between 0 and 1
        x = custom_sigmoid(x)  # To constrain values between 0 and 1
        return x


num_epochs = 10
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
   
# Device selection (CUDA GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

json_path = '/home/ps332/myViT/data/mpii_data.json'
image_dir = '/home/ps332/myViT/data/mpii_data/images/images'
# Define any transforms you want to apply to your images
# For example, normalization as used in your model
transforms = Compose([
    Resize((224, 224)),  # Resize the image
    ToTensor(),  # Convert the image to a PyTorch tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
])

def calculate_accuracy(predicted_keypoints, gt_keypoints, threshold=0.50):
    """
    Calculate the percentage of keypoints that are correctly predicted within a given threshold.

    :param predicted_keypoints: Predicted keypoints.
    :param gt_keypoints: Ground truth keypoints.
    :param threshold: Distance threshold to consider a prediction as correct.
    :return: Percentage of correctly predicted keypoints.
    """
    # Calculate the L2 distance (Euclidean) between predicted and ground truth keypoints
    distances = torch.sqrt(torch.sum((predicted_keypoints - gt_keypoints) ** 2, dim=-1))

    # A keypoint is correct if the distance to the ground truth is less than the threshold
    correct = distances < threshold

    # Calculate the percentage of correct keypoints
    accuracy = torch.mean(correct.float()) * 100  # Convert to percentage
    return accuracy.item()

# Instantiate the dataset
dataset = PoseEstimationDataset(
    json_path=json_path,
    image_dir=image_dir,
    transform=transforms
)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
model = PoseEstimationModule(num_keypoints=16, max_people=13).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss() # for keypoints regression
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    total_PCKh_score = 0.0
    num_samples = 0

# Inside your training loop
    for idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        images = data[0].to(device)
        keypoints = data[1].to(device) 
        #print("gt_keypoints" , keypoints)
        
        #denormalized_keypoints = data[2].to(device)  # Denormalized keypoints
        #print("denormalized_keypoints" , denormalized_keypoints)
    
        optimizer.zero_grad()

        predicted_keypoints = model(images)
        #print("predicted_keypoints" , predicted_keypoints)

        accuracy = calculate_accuracy(predicted_keypoints, keypoints)
        print(f"Batch {idx} - Accuracy: {accuracy}%")

        # Compute loss for keypoints with denormalized ground truth
        #loss_keypoints = criterion(predicted_keypoints, denormalized_keypoints)
    
        # compute loss for normalized keypoints
        loss = criterion(predicted_keypoints, keypoints)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        running_train_loss += loss.item()

        # Plot and save the training images with keypoints
        #if idx % 10 == 0:  # Adjust the frequency as needed
        #    plot_keypoints(images[0], keypoints[0], predicted_keypoints[0], epoch, idx, 'train')

    
    # Calculate and print the average training loss after all batches
    avg_train_loss = running_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}")
    



# Validation step
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, keypoints = data[:2]
            images, keypoints = images.to(device), keypoints.to(device)
            #print("gt_keypoints" , keypoints)


            predicted_keypoints = model(images)
            #accuracy = calculate_accuracy(predicted_keypoints, keypoints)
            #print(f"Batch {idx} - Accuracy: {accuracy}%")

            #print("predicted_keypoints" , predicted_keypoints)

            loss = criterion(predicted_keypoints, keypoints)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

       # Update the learning rate
    lr_scheduler.step()

