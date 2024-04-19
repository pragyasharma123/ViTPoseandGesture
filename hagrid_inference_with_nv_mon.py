import argparse
import logging
import time

import torch
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
from albumentations.pytorch import ToTensorV2
import albumentations as A

from constants import targets
from custom_utils.utils import build_model
from nv_mon import GPUMonitor

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

class Demo:
    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(classifier, transform, num_samples=100, device='cpu'):
        dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed_input = transform(image=dummy_input)['image'].unsqueeze(0)
        input_tensor = processed_input.repeat(num_samples, 1, 1, 1).to(device)

        gpu_monitor = GPUMonitor(0)  # Assuming the GPU index 0 is used.
        records = gpu_monitor.start_monitoring()

        start_time = time.time()
        with torch.no_grad():
            output = classifier(input_tensor)
        total_time = time.time() - start_time

        gpu_monitor.stop_monitoring()
        gpu_records = gpu_monitor.get_data(records)

        avg_inference_time = total_time / num_samples
        inferences_per_second = num_samples / total_time

        print(f"Avg inference time: {avg_inference_time:.3f} seconds per image")
        print(f"Inference per second: {inferences_per_second:.2f} inferences per second")

        return avg_inference_time, inferences_per_second, gpu_records

def parse_arguments(params=None):
    parser = argparse.ArgumentParser(description="Demo full frame classification...")
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")
    parser.add_argument("--output_json", type=str, default=None, help="The file path for where to output a json of the benchmark results.")
    return parser.parse_args(params)

def main():
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(conf).to(device)
    if conf.model.checkpoint:
        snapshot = torch.load(conf.model.checkpoint, map_location=device)
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()

    transform = Demo.get_transform_for_inf(conf.test_transforms)
    avg_time, ips, gpu_records = Demo.run(model, transform, 100, device)

    output_data = {
        "avg_inference_time": avg_time,
        "inferences_per_second": ips,
        "gpu_usage_records": gpu_records
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
