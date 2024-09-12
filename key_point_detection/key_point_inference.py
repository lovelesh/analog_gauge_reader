from PIL import Image

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

    def predict_heatmaps(self, image):
        """Key Point Predictor function

        Args:
            image (2D array): Cropped image of the Gauge

        Returns:
            np.array: Predicted heatmaps from the image
        """
        img = Image.fromarray(image)
        image_t = custom_transforms(train=False, image=img)
        image_t = image_t.unsqueeze(0)

        heatmaps = self.model(image_t)

        heatmaps = heatmaps.detach().squeeze(0).numpy()

        return heatmaps


def detect_key_points(heatmaps):
    """This function detects key points from the heatmap

    Args:
        heatmaps (np.array): Heatmap generated from the key point detector

    Returns:
        list: List of keypoints
    """
    key_point_list = full_key_point_extraction(heatmaps, 0.6)

    return key_point_list
