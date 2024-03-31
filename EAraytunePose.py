import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import os
import torch
from PIL import Image
import json
import random
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import time
import PoseUtility
import PoseEstimationDataset


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
# Configuration and global settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 19  # Update as necessary
num_keypoints = 16  # Update as necessary
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


def measure_inference_time(model, input_tensor, num_iterations=100):
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(input_tensor)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_iterations
    return avg_inference_time

def estimate_memory_traffic(model, input_tensor):
    total_memory_read = 0
    total_memory_write = 0

    bytes_per_element = 4  

    def compute_memory(tensor_size):
        return torch.prod(torch.tensor(tensor_size)).item() * bytes_per_element

    total_memory_read += compute_memory(input_tensor.size())

    def forward_hook(module, input, output):
        nonlocal total_memory_read, total_memory_write

        # Compute memory for inputs
        input_memory = sum(compute_memory(inp.size()) for inp in input if torch.is_tensor(inp))
        total_memory_read += input_memory

        # Handle complex output structures
        output_memory = 0
        if isinstance(output, torch.Tensor):
            output_memory = compute_memory(output.size())
        elif hasattr(output, 'last_hidden_state'):  # For models returning BaseModelOutput or similar
            output_memory = compute_memory(output.last_hidden_state.size())
        elif isinstance(output, tuple):
            for out in output:
                if torch.is_tensor(out):
                    output_memory += compute_memory(out.size())
        total_memory_write += output_memory

        # Account for parameters
        if hasattr(module, 'weight') and module.weight is not None:
            total_memory_read += compute_memory(module.weight.size())
        if hasattr(module, 'bias') and module.bias is not None:
            total_memory_read += compute_memory(module.bias.size())

    hooks = []
    for name, layer in model.named_modules():
        hooks.append(layer.register_forward_hook(forward_hook))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return total_memory_read + total_memory_write

# FLOPs Calculation for Convolutional Layers
def conv2d_flops(C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride=1, padding=0):
    # Assuming the output height and width can be calculated as follows:
    # out_height = (H_in + 2 * padding - kernel_size) // stride + 1
    # out_width = (W_in + 2 * padding - kernel_size) // stride + 1
    # Calculating the FLOPs for Conv2D
    flops_per_instance = kernel_size * kernel_size * C_in * C_out * H_out * W_out
    total_flops = flops_per_instance
    return total_flops


# FLOPs Calculation for Linear Layers
def linear_flops(input_features, output_features):
    # Calculating the FLOPs
    return input_features * output_features

# FLOPs Calculation for Layer Normalization
def layernorm_flops(tensor_size):
    return tensor_size * 2

# FLOPs Calculation for GELU Activation
def gelu_flops(tensor_size):
    return tensor_size * 8  

# FLOPs Calculation for Self-Attention
def self_attention_flops(seq_length, d_model):
    # Considering Q, K, V matrix multiplications and softmax computation
    flops_qkv = 3 * seq_length * d_model * d_model  # Q, K, V computation
    flops_softmax = seq_length * seq_length * d_model  # Softmax computation
    flops_output = seq_length * d_model * d_model  # Output computation
    return flops_qkv + flops_softmax + flops_output


def get_flops(model):
    def hook(module, input, output):
        module_class_name = str(module.__class__).split(".")[-1].split("'")[0]
        if isinstance(module, torch.nn.Conv2d):
        # Extracting parameters for Conv2d layer
            batch_size, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()
        
            kernel_height, kernel_width = module.kernel_size
            flops = batch_size * output_channels * output_height * output_width * (input_channels * kernel_height * kernel_width + 1)  # +1 for bias
            if hasattr(module, '__flops__'):
                module.__flops__ += flops
            else:
                module.__flops__ = flops
        
        elif isinstance(module, torch.nn.Linear):
        # Extracting parameters for Linear layer
            input_features = module.in_features
            output_features = module.out_features
            flops = input_features * output_features + output_features  # +output_features for bias
            if hasattr(module, '__flops__'):
                module.__flops__ += flops
            else:
                module.__flops__ = flops
        
        elif isinstance(module, torch.nn.LayerNorm):
            input_tensor = input[0]
            tensor_size = input_tensor.nelement()
            flops = layernorm_flops(tensor_size)
            if hasattr(module, '__flops__'):
                module.__flops__ += flops
            else:
                module.__flops__ = flops
        
        elif isinstance(module, torch.nn.GELU):
            input_tensor = input[0]
            tensor_size = input_tensor.nelement()
            flops = gelu_flops(tensor_size)
            if hasattr(module, '__flops__'):
                module.__flops__ += flops
            else:
                module.__flops__ = flops
        
        elif hasattr(module, 'is_attention'):
            seq_length, d_model = input[0].size(1), input[0].size(2)
            flops = self_attention_flops(seq_length, d_model)
            if hasattr(module, '__flops__'):
                module.__flops__ += flops
            else:
                module.__flops__ = flops

    def register_hooks(mod):
        if len(list(mod.children())) > 0 or isinstance(mod, torch.nn.Sequential):
            return
        if isinstance(mod, torch.nn.MultiheadAttention) or hasattr(mod, 'is_attention'):
            mod.is_attention = True
        mod.register_forward_hook(hook)
        mod.__flops__ = 0

    # Register hook for each module
    model.apply(register_hooks)

    # Dummy forward pass to trigger hooks
    input = torch.rand(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        model(input)

    total_flops = sum([mod.__flops__ for mod in model.modules() if hasattr(mod, '__flops__')])

    # Cleanup
    for mod in model.modules():
        if hasattr(mod, '__flops__'):
            del mod.__flops__

    return total_flops

def calculate_loss(model_accuracy, latency, flops, memory_traffic, k=1e-9):
    """
    :param model_accuracy: The accuracy of the model.
    :param latency: The latency of the model in seconds.
    :param flops: The floating-point operations per second.
    :param memory_traffic: The memory traffic in bytes.
    :param k: A scaling factor for the flops and memory traffic.
    :return: The loss.
    """
    # Assuming sota_accuracy is a predefined constant
    sota_accuracy = 0.99  
    
    # Calculate the loss based on the deviation from sota_accuracy and other factors
    loss = (sota_accuracy - model_accuracy) * latency * max(flops * k, memory_traffic)
    
    return loss

def calculate_distances(pred_keypoints, gt_keypoints):
    """
    Calculate distances between predicted keypoints and ground truth keypoints using broadcasting.
    Handles variable numbers of keypoints by operating on tensors.

    :param pred_keypoints: Predicted keypoints as a tensor of shape (num_predictions, 2).
    :param gt_keypoints: Ground truth keypoints as a tensor of shape (num_gt_keypoints, 2).
    :return: A tensor of distances of shape (num_predictions, num_gt_keypoints).
    """
    # Ensure pred_keypoints and gt_keypoints are on the same device
    pred_keypoints = pred_keypoints.to(device)
    gt_keypoints = gt_keypoints.to(device)

    # Expand dimensions to support broadcasting
    pred_keypoints_exp = pred_keypoints.unsqueeze(1)  # Shape: (num_predictions, 1, 2)
    gt_keypoints_exp = gt_keypoints.unsqueeze(0)  # Shape: (1, num_gt_keypoints, 2)

    # Calculate squared differences
    diff = pred_keypoints_exp - gt_keypoints_exp  # Broadcasting here
    distances = torch.sqrt(torch.sum(diff ** 2, dim=2))

    return distances

def match_keypoints(distances, threshold=0.05):
    """
    Match predicted keypoints to ground truth keypoints based on minimum distance, using a threshold.

    :param distances: A tensor of distances between predictions and ground truth keypoints, shape (num_predictions, num_gt_keypoints).
    :param threshold: Distance threshold for valid matches.
    :return: A list of tuples, where each tuple contains (pred_index, gt_index) for matched keypoints within the threshold.
    """
    matched_indices = []

    # Apply threshold
    valid_distances = distances < threshold

    # Find matches
    for gt_index in range(distances.shape[1]):  # Iterate over ground truth keypoints
        if valid_distances[:, gt_index].any():
            pred_index = torch.argmin(distances[:, gt_index])
            if distances[pred_index, gt_index] < threshold:
                matched_indices.append((pred_index.item(), gt_index))

    return matched_indices


def train_pose_estimation(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)

    # Prepare model
    num_layers = config["pose_num_layers"]
    dropout_rate = config["pose_dropout_rates"]

    model = CombinedModel(
        num_classes=len(class_names),
        num_keypoints=16,  
        max_people=13,
        gesture_layer_sizes=[1024, 512],  # Assuming these are fixed for this example
        gesture_dropout_rates=[0.5, 0.4],  # Assuming these are fixed for this example
        gesture_activations=["ReLU", "LeakyReLU"],  # Assuming these are fixed for this example
        pose_layer_sizes=config["pose_layer_sizes"],
        pose_dropout_rates=config["pose_dropout_rates"],
        pose_activations=config["pose_activations"]
)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)  # Move your model to the appropriate device

    # Prepare optimizer
    optimizer_type = config["optimizer_type"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get("optimizer_momentum", 0.9),  # Default to a common momentum value if not specified
            weight_decay=weight_decay
        )
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == "Adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get("optimizer_momentum", 0.9),  # Default to a common momentum value if not specified
            weight_decay=weight_decay
        )
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


    # Define any transforms you want to apply to your images
    p_transforms = Compose([
        Resize((224, 224)),  # Resize the image
        ToTensor(),  # Convert the image to a PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])


    def p_collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        keypoints_tensors = [item[1] for item in batch]  # Assuming these are your varying-size tensors
        # Pad sequences to match the longest sequence in the batch
        keypoints_tensors_padded = pad_sequence(keypoints_tensors, batch_first=True, padding_value=0)
        # Other items remain unchanged
        denormalized_keypoints_tensors = [item[2] for item in batch]  # Assuming these need similar treatment
        denormalized_keypoints_tensors_padded = pad_sequence(denormalized_keypoints_tensors, batch_first=True, padding_value=0)
        image_filenames = [item[3] for item in batch]
        orig_widths = torch.tensor([item[4] for item in batch])
        orig_heights = torch.tensor([item[5] for item in batch])

        return images, keypoints_tensors_padded, denormalized_keypoints_tensors_padded, image_filenames, orig_widths, orig_heights

    json_path = '/home/ps332/myViT/data/mpii_data.json'
    image_dir = '/home/ps332/myViT/data/mpii_data/images/images'

    # Instantiate the dataset
    dataset = PoseEstimationDataset.PoseEstimationDataset(
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

    for epoch in range(10):  
        model.train()
        total_train_loss = 0.0
        total_train_accuracy = 0.0

        for batch_idx, (images, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in enumerate(p_train_dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            output, _ = model(images)
            pred_keypoints_flat = output.view(output.size(0), -1, 2)  

            # Initialize batch loss as a tensor
            batch_loss = torch.tensor(0., device=device)
            batch_accuracy = 0.0 

            for i, gt_keypoints in enumerate(keypoints_tensor_list):
                gt_keypoints = gt_keypoints.to(device)
                distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                matched_indices = match_keypoints(distances, threshold=0.5)

                if not matched_indices:
                    print("No matched indices found, skipping...")
                    continue

                valid_predictions = [pred_keypoints_flat[i][mi[0]] for mi in matched_indices]
                gt_keypoints_subset = [gt_keypoints[mi[1]] for mi in matched_indices]

                valid_predictions = torch.stack(valid_predictions)
                gt_keypoints_subset = torch.stack(gt_keypoints_subset)

                if valid_predictions.dim() > gt_keypoints_subset.dim():
                    valid_predictions = valid_predictions.squeeze()

                keypoint_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints_subset)

                batch_loss += keypoint_loss
                batch_accuracy += PoseUtility.calculate_accuracy(valid_predictions, gt_keypoints_subset, threshold=0.05)

            # Perform backward pass and optimizer step only if there was at least one valid match
            if batch_loss.requires_grad:
                batch_loss.backward()
                optimizer.step()

            total_train_loss += batch_loss.item() if batch_loss.requires_grad else 0
            total_train_accuracy += batch_accuracy / max(1, len(keypoints_tensor_list))

            print(f"Batch {batch_idx + 1}, Loss: {batch_loss.item() if batch_loss.requires_grad else 0}, Accuracy: {batch_accuracy / max(1, len(keypoints_tensor_list)):.2f}%")

        avg_train_loss = total_train_loss / len(p_train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(p_train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}, Average Training Accuracy: {avg_train_accuracy:.2f}%")

        model.eval()
        total_val_loss = 0.0
        total_val_accuracy = 0.0

        with torch.no_grad():  # No need to track gradients for validation
            for batch_idx, (images, keypoints_tensor_list, denormalized_keypoints_list, image_filenames, orig_width, orig_height) in enumerate(p_val_dataloader):
                images = images.to(device)
                output, _ = model(images)
                pred_keypoints_flat = output.view(output.size(0), -1, 2)

                # Initialize batch loss as a tensor for consistency
                batch_loss = torch.tensor(0., device=device)
                batch_accuracy = 0.0

                for i, gt_keypoints in enumerate(keypoints_tensor_list):
                    gt_keypoints = gt_keypoints.to(device)
                    distances = calculate_distances(pred_keypoints_flat[i], gt_keypoints)
                    matched_indices = match_keypoints(distances, threshold=0.5)
        
                    if not matched_indices:
                        continue  # Skip if no matched indices

                    valid_predictions = [pred_keypoints_flat[i][mi[0]] for mi in matched_indices]
                    gt_keypoints_subset = [gt_keypoints[mi[1]] for mi in matched_indices]

                    valid_predictions = torch.stack(valid_predictions)
                    gt_keypoints_subset = torch.stack(gt_keypoints_subset)

                    if valid_predictions.dim() > gt_keypoints_subset.dim():
                        valid_predictions = valid_predictions.squeeze()  # Adjust dimensions if necessary

                    keypoint_loss = torch.nn.functional.mse_loss(valid_predictions, gt_keypoints_subset)
        
                    batch_loss += keypoint_loss
                    batch_accuracy += PoseUtility.calculate_accuracy(valid_predictions, gt_keypoints_subset, threshold=0.05)

                # Convert to item for scalar value
                batch_loss = batch_loss.item() if batch_loss.requires_grad else batch_loss
                batch_accuracy = batch_accuracy / max(1, len(keypoints_tensor_list))

                print(f"Batch {batch_idx + 1}, Loss: {batch_loss}, Accuracy: {batch_accuracy:.2f}%")

                total_val_loss += batch_loss
                total_val_accuracy += batch_accuracy

        avg_val_loss = total_val_loss / len(p_val_dataloader)
        avg_val_accuracy = total_val_accuracy / len(p_val_dataloader)
        train.report({"avg_val_accuracy": avg_val_accuracy})
        print(f"Validation - Epoch {epoch + 1}, Average Loss: {avg_val_loss:.4f}, Average Accuracy: {avg_val_accuracy:.2f}%")

        #input_tensor = torch.randn((1, 3, 224, 224)).to(device)  # Example input tensor
        #flops = get_flops(model) 

        #loss_function = calculate_loss(avg_val_accuracy*100, latency, flops, memory_traffic)

        #print(f"Loss calculated from loss function:", loss_function)


search_space = {
    # Pose Estimation Head parameters
    "pose_num_layers": tune.choice([2, 3, 4]),
    "pose_layer_sizes": tune.choice([(512, 256), (256, 128), (512, 512)]),
    "pose_dropout_rates": tune.choice([(0.5, 0.25), (0.4, 0.2)]),
    "pose_activations": tune.choice([("ReLU", "LeakyReLU"), ("ELU", "ELU")]),
    "weight_decay": tune.choice([0.0, 0.0001, 0.001, 0.01]),
    # Training parameters
    "batch_size": tune.choice([16, 32, 64]),
    "learning_rate": tune.choice([0.001, 0.0005, 0.0001]),
    "optimizer_type": tune.choice(['Adam', 'SGD']),
    
}

train_pose_estimation_with_resources = tune.with_resources(
    train_pose_estimation,
    resources={"cpu": 8, "gpu": 2}
)

algo  = OptunaSearch() 
algo = ConcurrencyLimiter(algo, max_concurrent=2)

num_samples = 100
tuner = tune.Tuner(train_pose_estimation, 
                   tune_config = tune.TuneConfig(
                       search_alg=algo,
                       metric="avg_val_accuracy",
                       mode="max",
                       num_samples=num_samples,
                   ),
                   param_space=search_space,
                   )
results = tuner.fit()


