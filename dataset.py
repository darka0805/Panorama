import os
import glob
import cv2
import geopandas as gpd

df = gpd.read_file('/kaggle/input/deforestation-in-ukraine/deforestation_labels.geojson')
base_dataset_path = '/kaggle/input/deforestation-in-ukraine/'
development_dataset_path = '/kaggle/working/smaller_dev_dataset/'

def preprocess_and_save_images(base_dataset_path, dev_dataset_path, max_dimension=1024):
    """
    Here we aim to get all images from our original folder to easier work with photos in the future
    """
    if not os.path.exists(dev_dataset_path):
        os.makedirs(dev_dataset_path)

    for folder in os.listdir(base_dataset_path):
        img_data_path = os.path.join(base_dataset_path, folder, "*.SAFE/GRANULE/*/IMG_DATA/*_TCI.jp2")
        tci_image_files = glob.glob(img_data_path)

        for image_file in tci_image_files:
            preprocessed_image = load_and_preprocess_image(image_file, max_dimension)
            dest_image_path = os.path.join(dev_dataset_path, os.path.basename(image_file).replace('.jp2', '.jpg'))

            # save the preprocessed image as a JPEG with quality set to 90, as with higher one it would take too much memory
            cv2.imwrite(dest_image_path, preprocessed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            print(f"Image saved at: {dest_image_path}")

def load_and_preprocess_image(image_path, max_dimension=1024):
    """
    here we perform resizing of an image
    """
    img = cv2.imread(image_path, 0)
    height, width = img.shape
    # here we calculate scaling factor to resize the image to the max_dimension
    scale = max_dimension / max(height, width)
    # here we perform resizing according to the scale
    resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return resized_img
preprocess_and_save_images(base_dataset_path, development_dataset_path)
