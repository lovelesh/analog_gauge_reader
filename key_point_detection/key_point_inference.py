from PIL import Image
import onnxruntime as ort
import numpy as np

from key_point_detection.key_point_extraction import full_key_point_extraction
from key_point_detection.model import load_model, load_optimized_model
from key_point_detection.key_point_dataset import custom_transforms


class KeyPointInference:
    def __init__(self, model_path, optimized=False):
        """Constructor to intialize the key point model

        Args:
            model_path (string): Path to the model
        """
        if not optimized:
            self.model = load_model(model_path)
        else:
            self.model = load_optimized_model(model_path)

    def predict_heatmaps(self, image, optimized=False):
        """Key Point Predictor function

        Args:
            image (2D array): Cropped image of the Gauge

        Returns:
            np.array: Predicted heatmaps from the image
        """
        img = Image.fromarray(image)
        image_t = custom_transforms(train=False, image=img)
        image_t = image_t.unsqueeze(0)
        print(f"image shape: {image_t.shape}")

        if not optimized:
            heatmaps = self.model(image_t)
            # print(f"heatmaps: {type(heatmaps)}")
            # print(f"heatmap shape: {heatmaps.shape}")
            heatmaps = heatmaps.detach().squeeze(0).numpy()
            # print(f"heatmap shape after: {heatmaps.shape}")
        else:
            ort_inputs = {self.model.get_inputs()[0].name: to_numpy(image_t)}
            heatmaps = self.model.run(None, ort_inputs)
            # print(f"heatmaps: {type(heatmaps)}")
            heatmaps = np.array(heatmaps)
            # print(f"heatmap shape: {heatmaps.shape}")
            heatmaps = heatmaps.squeeze()
            # print(f"heatmap shape after: {heatmaps.shape}")     

        return heatmaps

def detect_key_points(heatmaps):
    """This function detects key points from the heatmap

    Args:
        heatmaps (np.array): Heatmap generated from the key point detector

    Returns:
        list: List of keypoints
    """
    key_point_list = full_key_point_extraction(heatmaps, 0.6)
    print(f"key point list shape: {type(key_point_list)} {len(key_point_list)}")

    return key_point_list

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
