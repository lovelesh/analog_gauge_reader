import argparse
import os
import time
import sys

import matplotlib
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

# pylint: disable=wrong-import-position
from key_point_dataset import KeypointImageDataSet, \
    IMG_PATH, LABEL_PATH, TRAIN_PATH, RUN_PATH, custom_transforms
from key_point_extraction import full_key_point_extraction
from model import load_model, N_HEATMAPS

matplotlib.use('Agg')

HEATMAP_PREFIX = "H_"
KEY_POINT_PREFIX = "K_"

VAL_PATH = 'val'
TEST_PATH = 'test'


class KeyPointVal:
    def __init__(self, model, base_path, time_str=None):

        self.time_str = time_str if time_str is not None else time.strftime(
            "%Y%m%d-%H%M%S")

        train_image_folder = os.path.join(base_path, TRAIN_PATH, IMG_PATH)
        train_annotation_folder = os.path.join(base_path, TRAIN_PATH,
                                               LABEL_PATH)

        val_image_folder = os.path.join(base_path, VAL_PATH, IMG_PATH)
        val_annotation_folder = os.path.join(base_path, VAL_PATH, LABEL_PATH)

        self.base_path = base_path
        self.model = model

        self.train_dataset = KeypointImageDataSet(
            img_dir=train_image_folder,
            annotations_dir=train_annotation_folder,
            train=False,
            val=True)

        self.val_dataset = KeypointImageDataSet(
            img_dir=val_image_folder,
            annotations_dir=val_annotation_folder,
            train=False,
            val=True)

    def validate_set(self, path, dataset):
        for index, data in enumerate(dataset):
            print(index)
            image, original_image, annotation = data

            heatmaps = self.model(image.unsqueeze(0))
            print("inference done")
            # take it as numpy array and decrease dimension by one
            heatmaps = heatmaps.detach().numpy().squeeze(0)

            key_points = full_key_point_extraction(heatmaps, threshold=0.6)
            key_points_true = full_key_point_extraction(
                annotation.detach().numpy(), threshold=0.95)

            print("key points extracted")

            # plot the heatmaps in the run folder
            heatmap_file_path = os.path.join(
                path, HEATMAP_PREFIX + dataset.get_name(index) + '.jpg')
            plot_heatmaps(heatmaps, annotation, heatmap_file_path)
            key_point_file_path = os.path.join(
                path, KEY_POINT_PREFIX + dataset.get_name(index) + '.jpg')
            #resize original image as well
            original_image_tensor = custom_transforms(train=False,
                                                      image=original_image)
            plot_key_points(original_image_tensor, key_points, key_points_true,
                            key_point_file_path)

    def validate(self):
        run_path = os.path.join(self.base_path, RUN_PATH + '_' + self.time_str)
        train_path = os.path.join(run_path, TRAIN_PATH)
        val_path = os.path.join(run_path, VAL_PATH)

        os.makedirs(run_path, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        self.validate_set(train_path, self.train_dataset)
        self.validate_set(val_path, self.val_dataset)


def plot_heatmaps(heatmaps1, heatmaps2, filename):
    plt.figure(figsize=(12, 8))

    if N_HEATMAPS == 1:
        plt.subplot(2, 1, 1)
        plt.imshow(heatmaps1.squeeze(0), cmap=plt.cm.viridis)
        plt.title('Predicted Heatmap')

        plt.subplot(2, 1, 2)
        plt.imshow(heatmaps2.squeeze(0), cmap=plt.cm.viridis)
        plt.title('True Heatmap')

    else:
        titles = ['Start', 'Middle', 'End']

        for i in range(3):
            plt.subplot(2, 3, i + 1)
            plt.imshow(heatmaps1[i], cmap=plt.cm.viridis)
            plt.title(f'Predicted Heatmap {titles[i]}')

        for i in range(3):
            plt.subplot(2, 3, i + 4)
            plt.imshow(heatmaps2[i], cmap=plt.cm.viridis)
            plt.title(f'True Heatmap {titles[i]}')

    # Adjust the layout of the subplots
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')


def plot_key_points(image, key_points_pred, key_points_true, filename):
    plt.figure(figsize=(12, 8))

    image = image.permute(1, 2, 0)

    if N_HEATMAPS == 1:
        key_points = key_points_pred[0]
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title('Predicted Key Point')

        key_points = key_points_true[0]
        plt.subplot(2, 1, 2)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title('True Key Point')
    else:
        titles = ['Start', 'Middle', 'End']
        for i in range(3):
            key_points = key_points_pred[i]
            plt.subplot(2, 3, i + 1)
            plt.imshow(image)
            plt.scatter(key_points[:, 0],
                        key_points[:, 1],
                        s=50,
                        c='red',
                        marker='x')
            plt.title(f'Predicted Key Point {titles[i]}')

        for i in range(3):
            key_points = key_points_true[i]
            plt.subplot(2, 3, i + 4)
            plt.imshow(image)
            plt.scatter(key_points[:, 0],
                        key_points[:, 1],
                        s=50,
                        c='red',
                        marker='x')
            plt.title(f'True Key Point {titles[i]}')

    # Adjust the layout of the subplots
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')


def main():
    args = read_args()

    model_path = args.model_path
    base_path = args.data

    model = load_model(model_path)

    validator = KeyPointVal(model, base_path)
    validator.validate()


def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="path to pytorch model")
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help="Base path of data")

    return parser.parse_args()


if __name__ == '__main__':
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.append(parent_dir)
    main()
