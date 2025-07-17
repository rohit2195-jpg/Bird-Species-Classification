import os
import shutil
from collections import defaultdict
import random

def createFolders():
    key_file = open('other_datasets/nabirds/train_test_split.txt', 'r')
    key_lines = key_file.readlines()
    key_file.close()
    image_to_decision ={}
    number_to_class = {}
    classes = open('other_datasets/nabirds/classes.txt', 'r')
    class_lines = classes.readlines()

    test_dir = "./Test"
    train_dir = "./Train"
    source_dir = "./other_datasets/nabirds/images"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


    for line in class_lines:
        line = line.strip().split(" ", maxsplit=1)
        name = line[1].replace("/", "-")
        number_to_class[int(line[0])] = name

    for line in key_lines:
        line_content = line.strip().split(" ")
        name = line_content[0].replace("-","")
        image_to_decision[name] = line_content[1]

    for bird_folder in os.listdir(source_dir):
        try:
            bird_folder_name = number_to_class[int(bird_folder)]
        except:
            print("invalid folder ", bird_folder)
            continue

        for image in os.listdir(os.path.join(source_dir, bird_folder)):
            test_or_train = int(image_to_decision[image[0:image.index(".")]])
            if (test_or_train == 1):
                #training
                os.makedirs(os.path.join(train_dir, bird_folder_name), exist_ok=True)
                shutil.copy(os.path.join(source_dir, bird_folder, image), os.path.join(train_dir, bird_folder_name, image))
            elif (test_or_train == 0):
                os.makedirs(os.path.join(test_dir, bird_folder_name), exist_ok=True)
                shutil.copy(os.path.join(source_dir, bird_folder, image), os.path.join(test_dir, bird_folder_name, image))
    print("Data is seperated into test and train")

def combine_similar_folders(train_dir):
    # Dictionary to group folders by base name
    folder_groups = defaultdict(list)

    # Loop through all items in the train directory
    for folder_name in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder_name)

        # Only process directories
        if os.path.isdir(folder_path):
            # Get the base name (everything before the first parenthesis)
            base_name = folder_name.split('(')[0].strip()
            folder_groups[base_name].append(folder_path)

    # Combine folders with the same base name
    for base_name, paths in folder_groups.items():
        if len(paths) > 1:
            combined_folder = os.path.join(train_dir, base_name)
            os.makedirs(combined_folder, exist_ok=True)

            for path in paths:
                for file in os.listdir(path):
                    src_file = os.path.join(path, file)
                    # Avoid overwriting files with the same name
                    dst_file = os.path.join(combined_folder, file)
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(file)
                        i = 1
                        while os.path.exists(dst_file):
                            dst_file = os.path.join(combined_folder, f"{base}_{i}{ext}")
                            i += 1
                    shutil.move(src_file, dst_file)

                # Remove the now-empty folder
                os.rmdir(path)
def create_validation():
    validation = "./validation"
    for bird_folder in os.listdir("./Test"):
        source_folder = os.path.join("Test", bird_folder)
        dest_folder = os.path.join(validation, bird_folder)
        if not os.path.isdir(source_folder):
            continue
        os.makedirs(dest_folder, exist_ok=True)

        images = os.listdir(source_folder)
        random.shuffle(images)
        half = len(images) // 2

        for image in images[:half]:
            source_file = os.path.join(source_folder, image)
            shutil.move(source_file, dest_folder)


train_folder_path = "Test"  # Change this if your path is different
create_validation()









