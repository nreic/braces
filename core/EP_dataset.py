import random
import shutil
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
from pprint import pprint

teeth_class_label = 8
highlight_color = (0, 255, 0)  # green in BGR


def make_dataset(original_images_file, root_dir="../data/dataset", train_size=300, test_size=100, val_size=100):
    subdirs = ['train', 'test', 'val']
    subsubdirs = ['images', 'annotations']
    directories = [
        os.path.join(root_dir, subdir, subsubdir)
        for subdir in subdirs
        for subsubdir in subsubdirs
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("New dataset folders created in", root_dir)

    with open(original_images_file, 'r') as file:
        orig_images = file.read().splitlines()
    random.shuffle(orig_images)

    img_source_folder = os.path.join(EP_dataset_dir, 'images', 'train')
    annot_source_folder = os.path.join(EP_dataset_dir, 'annotations', 'train')

    copy_images(orig_images[:train_size],
                img_source_folder,
                annot_source_folder,
                os.path.join(root_dir, 'train'))
    copy_images(orig_images[train_size:test_size + train_size],
                img_source_folder,
                annot_source_folder,
                os.path.join(root_dir, 'test'))
    copy_images(orig_images[train_size + test_size: train_size + test_size + val_size],
                img_source_folder,
                annot_source_folder,
                os.path.join(root_dir, 'val'))

    print("New dataset ready to use.")


def copy_images(images, img_source_folder, annot_source_folder, dest_base_folder):
    img_dest = os.path.join(dest_base_folder, 'images')
    annot_dest = os.path.join(dest_base_folder, 'annotations')
    for img_name in images:
        img_path = os.path.join(img_source_folder, img_name+'.jpg')
        shutil.copy(img_path, img_dest)
        annot_path = os.path.join(annot_source_folder, img_name+'.png')
        shutil.copy(annot_path, annot_dest)
    print(f"Images copied from {img_source_folder} to {dest_base_folder}.")


def extract_teeth_images_from_folder(folder):
    img_source_folder = os.path.join(EP_dataset_dir, 'images', folder)
    annot_source_folder = os.path.join(EP_dataset_dir, 'annotations', folder)
    os.makedirs(f"../data/EasyPortrait-dataset/dataset_teeth_all_{folder}/images",
                exist_ok=True)
    os.makedirs(f"../data/EasyPortrait-dataset/dataset_teeth_all_{folder}/annotations",
                exist_ok=True)
    with open(f"../data/images_with_teeth_from_{folder}", 'r') as file:
        orig_images = file.read().splitlines()
    copy_images(orig_images, img_source_folder,
                annot_source_folder,
                f"../data/EasyPortrait-dataset/dataset_teeth_all_{folder}")


def copy_annotations_of_images(image_dir, annot_src_dir):
    annot_dest_dir = os.path.join(image_dir, '../..', 'annotations')
    #os.makedirs(annot_dest_dir, exist_ok=True)
    images = os.listdir(image_dir)
    for img_name in images:
        for folder in ["train", "test", "val"]:
            annot_src_path = os.path.join(annot_src_dir, folder, img_name.replace('.jpg', '.png'))
            if os.path.isfile(annot_src_path):
                shutil.copy(annot_src_path, annot_dest_dir)
                break


def find_all_teeth_img():
    folder = "train"
    EP_annot_dir = os.path.join(EP_dataset_dir, "/annotations", folder)
    annot_files = os.listdir(EP_annot_dir)

    img_with_teeth = []
    for annot_file in annot_files:
        annot_file_path = os.path.join(EP_annot_dir, annot_file)
        annot_img = cv2.imread(annot_file_path)
        if np.any(annot_img[:, :, 0] == teeth_class_label):
            img_with_teeth.append(annot_file.split('.')[0])

    EP_teeth_images = f"images_with_teeth_from_{folder}"
    write_list_to_file(img_with_teeth, EP_teeth_images)


def write_list_to_file(file_list, out_path):
    with open(out_path, 'w') as file:
        for element in file_list:
            file.write(element + '\n')


def analyse_img(img):
    img = cv2.imread(get_img_path(img))
    plt.imshow(img, cmap='gray')
    plt.hist(img.flat, bins=200, range=(0,255))
    plt.show()


def get_img_path(img_name):
    return os.path.join(dataset_dir, folder, "images", img_name)
def get_annot_path(img_name):
    return os.path.join(dataset_dir, folder, "annotations",
                        img_name.replace('.jpg', '.png'))


def get_image_meta_infos():
    image_sizes = dict()
    for img in images_files:
        img_path = get_img_path(img)
        image = cv2.imread(img_path)
        dimensions = image.shape
        image_sizes[dimensions] = image_sizes.get(dimensions, 0) + 1
    image_sizes = {k: int(v) / len(images_files)
                   for k,v in image_sizes.items()}
    return image_sizes


def create_meta_file(metafile="meta.csv"):
    meta_infos = []
    for img in images_files:
        img_path = get_img_path(img)
        image = cv2.imread(img_path)
        meta_infos.append({
            "file_name": img,
            "folder": folder,
            "height": image.shape[0],
            "width": image.shape[1]
        })
    pd.DataFrame(meta_infos).to_csv(metafile, index=False, header=True)


def visualize_teeth_distribution(out_dimensions=(128, 128)):
    teeth_distribution = np.zeros(out_dimensions)
    for annot in annotations_files:
        annot_path = get_annot_path(annot)
        annot_img = cv2.imread(annot_path, 0)
        resized_annot_img = cv2.resize(annot_img, (128,128))
        # teeth_distribution[:annot_img.shape[0],:annot_img.shape[1]][annot_img == teeth_class_label] += 1
        teeth_distribution[resized_annot_img == teeth_class_label] += 1
    teeth_distribution /= len(annotations_files)
    max_value = np.max(teeth_distribution)
    #teeth_distribution[:,1440:1443] = max_value
    plt.figure(dpi=300)
    plt.imshow(teeth_distribution, cmap="gray", vmin=0, vmax=max_value)
    plt.savefig("../outputs/output/teeth_distr_resized_128.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    EP_dataset_dir = '../../data/EasyPortrait-dataset'
    dataset_dir = '../../data/dataset_2_n300'
    folder = "train"

    images_path = os.path.join(dataset_dir, folder, 'images')
    images_files = os.listdir(images_path)
    annotations_path = os.path.join(dataset_dir, folder, 'annotations')
    annotations_files = os.listdir(annotations_path)

    visualize_teeth_distribution()
    #create_meta_file()
    #pprint(get_image_sizes())

    #analyse_img(test_img)
    #find_all_teeth_img()

    #make_dataset("images_with_teeth_from_train", "dataset_n500")
    #copy_annotations_of_images("dataset_2_n300/val/images", "../EasyPortrait-dataset/annotations/")
    #extract_teeth_images_from_folder('val')
