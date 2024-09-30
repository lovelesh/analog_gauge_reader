from PIL import Image
import onnxruntime as ort
import numpy as np
import random

from torchvision import transforms
import torchvision.transforms.functional as TF

from key_point_detection.key_point_extraction import full_key_point_extraction

INPUT_SIZE = (448, 448)
N_HEATMAPS = 3

class KeyPointInference:
    def __init__(self, model_path):
        """Constructor to intialize the key point model

        Args:
            model_path (string): Path to the model
        """
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
        # print(f"image shape: {image_t.shape}")

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
    # print(f"key point list shape: {type(key_point_list)} {len(key_point_list)}")

    return key_point_list

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_optimized_model(model_path):
    model = ort.InferenceSession(model_path, providers=ort.get_available_providers())
    return model

def custom_transforms(train, image, annotation=None, debug=False):

    resize = transforms.Resize(INPUT_SIZE,
                               transforms.InterpolationMode.BILINEAR)
    image = resize(image)
    if annotation is not None:
        annotation = resize(annotation)

    toTensor = transforms.ToTensor()
    # random crop image and annotation
    if train:
        if random.random() > 0.1:

            if random.random() > 0:
                angle = random.randint(-180, 180)
                image = TF.rotate(image, angle)
                annotation = TF.rotate(annotation, angle)

                if debug:
                    _plot_annotation_image(image, annotation)

            new_size = int(1.2 * INPUT_SIZE[0])
            resize = transforms.Resize(new_size,
                                       transforms.InterpolationMode.BILINEAR)
            # increase size
            image = resize(image)
            annotation = resize(annotation)

            top = random.randint(0, new_size - INPUT_SIZE[0])
            left = random.randint(0, new_size - INPUT_SIZE[0])
            image = TF.crop(image, top, left, INPUT_SIZE[0], INPUT_SIZE[1])
            annotation = TF.crop(annotation, top, left, INPUT_SIZE[0],
                                 INPUT_SIZE[1])

            if debug:
                _plot_annotation_image(image, annotation)

            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)

                if debug:
                    _plot_annotation_image(image, annotation)

    if annotation is not None:
        return toTensor(image), toTensor(annotation)
    return toTensor(image)


def _plot_annotation_image(image, annotation):
    image_np = np.asarray(image)
    annotation_np = np.asarray(annotation)
    mask = np.max(annotation_np, axis=2) < 0.99
    mask = np.stack([mask] * 3, axis=-1)
    merge = np.where(mask, image_np, annotation_np)
    merge_img = Image.fromarray(merge)
    image.show()
    merge_img.show()

