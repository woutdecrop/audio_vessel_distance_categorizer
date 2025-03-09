import os
import random
from tqdm import tqdm
from pydub import AudioSegment

def calculator(filename):
    parts = filename.split('_')
    last_part = parts[-1].split('.')[0]  # Remove the ".wav" extension
    number = int(last_part)
    
    if number > 10000:
        number = 10000
    
    result = (10000 - number) / 10000
    return round(result,3)

# Function to create the text files
def create_txt_files(absence_boats_folder, train_txt_file, test_txt_file, val_txt_file,class_txt_file):
    file_paths = []
    for root, dirs, files in os.walk(absence_boats_folder):
        for file in tqdm(files, desc="Processing files",position=1, leave=True):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path,absence_boats_folder)
            relative_path=relative_path.replace(" ", "_")
            try:
                AudioSegment.from_file(file_path) 
                file_paths.append(relative_path)
            except:
                print("skipping")
                pass
    # Get a list of folder names within the "absence_boats" directory

    # Get a list of folder names within the "absence_boats" directory
    folder_names = next(os.walk(absence_boats_folder))[1]

    # Assign numbers based on the location of each folder in the list
    folder_numbers = {folder_names[i]: i for i in range(len(folder_names)) if folder_names[i] != ".ipynb_checkpoints"}

    folder_numbers = {}
    index_counter = 0

    for i in range(len(folder_names)):
        if folder_names[i] != ".ipynb_checkpoints":
            folder_numbers[folder_names[i]] = index_counter
            index_counter += 1

    # Split the boat files into training, testing, and validation sets
    random.shuffle(file_paths)
    num_samples = len(file_paths)
    train_cutoff = int(num_samples * train_ratio)
    test_cutoff = train_cutoff + int(num_samples * test_ratio)

    train_files = [file for file in file_paths if any(subfolder in file for subfolder in ['train'])]
    test_files = [file for file in file_paths if any(subfolder in file for subfolder in ['test'])]
    val_files = [file for file in file_paths if any(subfolder in file for subfolder in ['val'])]

    # Create the training text file
    with open(train_txt_file, 'w') as f_train:
        for file in tqdm(train_files, desc="Writing training file"):
            file = file.replace('\\', '/')
            f_train.write(file + ' ' + str(calculator(file)) + '\n')

    # Create the testing text file
    with open(test_txt_file, 'w') as f_test:
        for file in tqdm(test_files, desc="Writing testing file"):
            file = file.replace('\\', '/')
            f_test.write(file + ' ' + str(calculator(file)) + '\n')

    # Create the validation text file
    with open(val_txt_file, 'w') as f_val:
        for file in tqdm(val_files, desc="Writing validation file"):
            file = file.replace('\\', '/')
            f_val.write(file + ' ' + str(calculator(file)) + '\n')

    # Create the classes text file
    with open(class_txt_file, 'w') as f_class:
        for label in tqdm(folder_numbers, desc="Writing classes file"):
            f_class.write(str(label) + '\n')

# Set the train, test, and validation ratios
train_ratio = 0.9
test_ratio = 0.1
val_ratio = 0
absence_boats_folder = 'C:/Users/wout.decrop/project_environment/module_imagine/underwater-noise-classification/data/audios'
data_set_folder = 'C:/Users/wout.decrop/project_environment/module_imagine/underwater-noise-classification/data/dataset_files'

absence_boats_folder = '/srv/underwater-noise-classification/data/audios'
data_set_folder = '/srv/underwater-noise-classification/data/dataset_files'

absence_boats_folder = '/storage/Imagine_UC6_new/DATA/data_per_station_10_updated_metadata_extra_filter'
# data_set_folder = '/storage/Imagine_UC6/data_new_ais/dataset_files'
data_set_folder = '/srv/underwater-noise-classification/data/dataset_files'
train_txt_file=os.path.join(data_set_folder,'train.txt')
test_txt_file =os.path.join(data_set_folder,'test.txt')
val_txt_file=os.path.join(data_set_folder,'val.txt')
class_txt_file=os.path.join(data_set_folder,'classes.txt')
# Get the relative file paths of all files within the "boats" and "no_boats" subfolders


# Create the text files
create_txt_files(absence_boats_folder,train_txt_file, test_txt_file, val_txt_file,class_txt_file)