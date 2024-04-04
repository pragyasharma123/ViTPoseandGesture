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
