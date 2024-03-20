import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import os
import torch
from PIL import Image
import json
import random
from typing import Tuple
from torch.utils.data import DataLoader
import pandas as pd
import time

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

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size=768, num_classes=18, fc1_size=1024, fc2_size=18, dropout_rate=0.5):
        super(GestureRecognitionHead, self).__init__()
        self.fc1 = nn.Linear(embedding_size, fc1_size)
        self.fc1_bn = nn.BatchNorm1d(fc1_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13):
        super(PoseEstimationHead, self).__init__()
        self.max_people = max_people
        self.fc1 = nn.Linear(combined_feature_size, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_keypoints * max_people * 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        batch_size = x.size(0)
        num_keypoints = self.fc2.out_features // (self.max_people * 2)
        x = x.view(batch_size, self.max_people, num_keypoints, 2)
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, fc1_size=1024, fc2_size=num_classes, dropout_rate=0.5):
        super(CombinedModel, self).__init__()
        self.max_people = max_people

        
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, fc1_size=fc1_size, fc2_size=fc2_size, dropout_rate=dropout_rate)
        
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        hidden_size = self.backbone.config.hidden_size
        self.pose_estimation_head = PoseEstimationHead(combined_feature_size=512 + hidden_size, num_keypoints=num_keypoints, max_people=max_people)
        
        # Freeze the Pose Estimation Head
        for param in self.pose_estimation_head.parameters():
            param.requires_grad = False

          # Explicitly freeze the feature_extractor and cnn_feature_processor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in self.cnn_feature_processor.parameters():
            param.requires_grad = False    

    def forward(self, x):
       
        vit_outputs = self.backbone(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]
        gesture_output = self.gesture_head(vit_features)
        
        with torch.no_grad():
            cnn_features = self.feature_extractor(x)
            processed_cnn_features = self.cnn_feature_processor(cnn_features)
            combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)
            keypoints = self.pose_estimation_head(combined_features)
            keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range
        
            return keypoints, gesture_output

# Gesture Dataset
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

def g_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

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

def calculate_fitness(model_accuracy, latency, flops, memory_traffic, k=1e-9):
    """
    Calculate the fitness score for a model configuration.
    
    :param model_accuracy: The accuracy of the model.
    :param latency: The latency of the model in seconds.
    :param flops: The floating-point operations per second.
    :param memory_traffic: The memory traffic in bytes.
    :param k: A scaling factor for the flops and memory traffic.
    :return: The fitness score.
    """
    # Assuming sota_accuracy is a predefined constant
    sota_accuracy = 0.99  
    
    # Calculate the loss based on the deviation from sota_accuracy and other factors
    loss = (sota_accuracy - model_accuracy) * latency * max(flops * k, memory_traffic)
    
    # The fitness function could be defined as the inverse of loss
    fitness = 1 / loss
    return fitness


def train_gesture_classification(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 42
    num_epoch = 1
    num_classes = len(class_names)
    learning_rate = 0.005
    
    # Initialize your model, criterion, optimizer, and scheduler here using the config dict
    model = CombinedModel(num_classes=num_classes,
                          num_keypoints=16,
                          dropout_rate=config["dropout_rate"]).to(device)
    
    optimizer = getattr(torch.optim, config["optimizer_type"])(
        model.parameters(), lr=config["learning_rate"])
    
    
    batch_size = config.get("batch_size", 64)  # Dynamic batch size

    g_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0.00001)


    g_transform = Compose([
    Resize((224, 224)),  # Resize to a multiple of 14 for patch embedding compatibility
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

    
    # Load your data here
    train_data = GestureDataset(
        path_images='/home/ps332/myViT/subsample',
        path_annotation='/home/ps332/myViT/ann_subsample',
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

    torch.manual_seed(random_seed)
    random.seed(random_seed)
   
    g_train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=g_collate_fn, shuffle=True, num_workers=2, drop_last=True)
    g_test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=g_collate_fn, shuffle=False, num_workers=2, drop_last=True)

    for epoch in range(10):  
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in g_train_loader:  # Iterate over batches of the gesture training dataset
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()  # Zero the gradients before running the backward pass.
        
            # Forward pass through the gesture recognition head only
            _, gesture_outputs = model(images)  # Assuming your model returns pose and gesture outputs
        
            # Calculate loss only for gesture recognition
            g_loss = g_criterion(gesture_outputs, labels)
            print(f"Gesture Training Loss", g_loss)
            g_loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model parameters
        
            # Accumulate loss and accuracy
            train_loss += g_loss.item() * images.size(0)
            _, predicted = torch.max(gesture_outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            train_accuracy = train_correct / train_total
            print(f"Gesture train accuracy:" , train_accuracy * 100)
        
        # Compute epoch-level training loss and accuracy
        epoch_loss = train_loss / train_total
        epoch_accuracy = train_correct / train_total * 100
    
        # Optional: Print epoch-level training details
        print(f'Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%')
    
        # Update the learning rate scheduler
        g_scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():  # Turn off gradient computation for evaluation
            for i, (images, labels) in enumerate(g_test_loader):
                images = images.to(device)
                labels = labels.to(device)
        
                # Forward pass through the gesture recognition head only
                _, gesture_outputs = model(images)  # Assuming your model returns pose and gesture outputs

                # Measure inference time and memory traffic on the first batch
                if i == 0:
                    input_tensor = images[:1]  # Use the first image in the batch for measurement
                    latency = measure_inference_time(model, input_tensor)
                    memory_traffic = estimate_memory_traffic(model, input_tensor)
                
                # Calculate loss only for gesture recognition
                g_loss = g_criterion(gesture_outputs, labels)
                print(f"Gesture Validation Loss" , g_loss)
        
                # Accumulate loss and accuracy
                val_loss += g_loss.item() * images.size(0)
                _, predicted = torch.max(gesture_outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_accuracy = val_correct / val_total
                print(f"Gesture validation accuracy:" , val_accuracy * 100)
        
            # Compute epoch-level training loss and accuracy
            epoch_loss = val_loss / val_total
            epoch_accuracy = val_correct / val_total * 100

            print(f'Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%')
        
            input_tensor = torch.randn((1, 3, 224, 224)).to(device)  # Example input tensor
            flops = get_flops(model) 
            fitness = calculate_fitness(val_accuracy, latency, flops, memory_traffic)

            train.report({"accuracy": val_accuracy*100, "latency": latency, "memory_traffic": memory_traffic, "flops": flops, "fitness": fitness})


search_space = {
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
    "dropout_rate": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "optimizer_type": tune.choice(['Adam', 'SGD', 'RMSprop', 'AdamW', 'Adagrad']),
    "lr_scheduler_type": tune.choice(['StepLR', 'ExponentialLR']),
    "lr_scheduler_step_size": tune.choice([5, 10, 20]),
    "lr_scheduler_gamma": tune.choice([0.1, 0.5, 0.9]),
}

scheduler = ASHAScheduler(
    max_t=100,
    grace_period=10,
    reduction_factor=2)

search_alg = OptunaSearch(metric="accuracy", mode="max")

analysis = tune.run(
    tune.with_parameters(train_gesture_classification),
    resources_per_trial={"cpu": 8, "gpu": 4},
    config=search_space,
    num_samples=10,
    metric="accuracy",  
    mode="max",  
    scheduler=scheduler,
    search_alg=search_alg)

print("Best hyperparameters found were: ", analysis.best_config)