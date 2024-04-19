import argparse
import logging
import time

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch import ToTensorV2
import albumentations as A

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

class Demo:
    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(classifier, transform, num_samples=100) -> None:
        """
        Run model inference on dummy data and measure performance
        Parameters
        ----------
        classifier : Model
            Classifier model
        transform :
            Albumentations transform
        num_samples : int
            Number of dummy samples to process for benchmarking
        """
        # Create a dummy input image of the same size expected by the model (e.g., 3x224x224)
        dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed_input = transform(image=dummy_input)['image'].unsqueeze(0)  # Add batch dimension

        # Instead of preparing a list of tensors, directly duplicate the tensor to simulate batch processing
        input_tensor = processed_input.repeat(num_samples, 1, 1, 1)  # Simulate a batch of the same image

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = classifier(input_tensor)  # Directly pass the batched tensor
        total_time = time.time() - start_time

        avg_inference_time = total_time / num_samples
        inferences_per_second = num_samples / total_time

        print(f"Avg inference time: {avg_inference_time:.3f} seconds per image")
        print(f"Inference per second: {inferences_per_second:.2f} inferences per second")

def parse_arguments(params=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification...")
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")
    return parser.parse_args(params)

if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    model = build_model(conf)
    transform = Demo.get_transform_for_inf(conf.test_transforms)
    if conf.model.checkpoint:
        snapshot = torch.load(conf.model.checkpoint, map_location="cpu")
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()
    Demo.run(model, transform)
