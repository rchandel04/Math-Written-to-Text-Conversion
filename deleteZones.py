import os
from sklearn.model_selection import train_test_split
import shutil

def split_dataset(root_folder, output_folder, test_size=0.2, random_seed=42):
    # Get the list of class folders
    class_folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]

    # Create output folders for training and testing
    train_output_folder = os.path.join(output_folder, 'train')
    test_output_folder = os.path.join(output_folder, 'test')

    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)

    # Loop through each class folder
    for class_folder in class_folders:
        class_path = os.path.join(root_folder, class_folder)

        # Get the list of images in the class folder
        images = [image for image in os.listdir(class_path) if image.endswith(('.jpg', '.jpeg', '.png'))]

        # Split the images into training and testing sets
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_seed)

        # Create class folders in the output directories
        train_class_output_folder = os.path.join(train_output_folder, class_folder)
        test_class_output_folder = os.path.join(test_output_folder, class_folder)

        os.makedirs(train_class_output_folder, exist_ok=True)
        os.makedirs(test_class_output_folder, exist_ok=True)

        # Copy training images to the training folder
        for image in train_images:
            source_path = os.path.join(class_path, image)
            destination_path = os.path.join(train_class_output_folder, image)
            shutil.copyfile(source_path, destination_path)

        # Copy testing images to the testing folder
        for image in test_images:
            source_path = os.path.join(class_path, image)
            destination_path = os.path.join(test_class_output_folder, image)
            shutil.copyfile(source_path, destination_path)

if __name__ == "__main__":
    # Set the path to your dataset folder
    dataset_path = 'extracted_images'

    # Set the path to the output folder
    output_path = 'data'

    # Specify the test size and random seed
    test_size = 0.2
    random_seed = 42

    # Call the function to split the dataset
    split_dataset(dataset_path, output_path, test_size, random_seed)
