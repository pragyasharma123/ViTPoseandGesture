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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


class PoseEstimationDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        #self.resize = Resize(self.target_size)

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

#        image = self.resize(image)
        if self.transform:
            image = self.transform(image)

        keypoints_tensor = torch.zeros((max_people, num_keypoints, 2))

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



num_keypoints = 16
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
        # Apply sigmoid to the output to constrain the values between 0 and 1
        x = torch.sigmoid(x)


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

# Inside your training loop
    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = data[0].to(device)
        keypoints = data[1].to(device)  # Normalized keypoints
        #print("gt_keypoints" , keypoints)
    
        #denormalized_keypoints = data[2].to(device)  # Denormalized keypoints
        #print("denormalized_keypoints" , denormalized_keypoints)

        optimizer.zero_grad()

        predicted_keypoints = model(images)
        #print("predicted_keypoints" , predicted_keypoints)

        # Compute loss for keypoints with denormalized ground truth
        #loss_keypoints = criterion(predicted_keypoints, denormalized_keypoints)
    
        # compute loss for normalized keypoints
        loss = criterion(predicted_keypoints, keypoints)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


        running_train_loss += loss.item()
    
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
            #print("predicted_keypoints" , predicted_keypoints)

            loss = criterion(predicted_keypoints, keypoints)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

       # Update the learning rate
    lr_scheduler.step()

