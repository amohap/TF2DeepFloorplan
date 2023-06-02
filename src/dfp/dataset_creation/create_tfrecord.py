import os
import random
from tf_record import *

train_file = 'structured3d_train.txt'
test_file = 'structured3d_test.txt'

# debug
if __name__ == '__main__':
    write_new_txt = False
    write_tf_record = True

    if write_new_txt:

        # write the r3d_train.txt and r3d_test.txt
        # Specify your directory
        directory = '/home/amohap/MT/code/t-mt-2023-FloorplanReconstruction-AdrianaMohap/source/trajectory_sampling/plots/tf2deep/'

        # List all files in directory
        all_files = os.listdir(directory)

        # Initialize dictionary to hold all scenes
        scenes = {}

        # Order of the files for a scene
        order = ['', '_wall', '_close', '_rooms', '_close_wall']

        # Go through each file
        for file_name in all_files:
            if file_name.endswith('.png'):
                # Extract the scene number (first 5 characters of the file name)
                scene_num = file_name[:5]

                # If the scene number is not in the dictionary yet, add an empty list for it
                if scene_num not in scenes:
                    scenes[scene_num] = [None]*5

                # Add the file name to the correct position in the scene's list in the dictionary
                for i, suffix in enumerate(order):
                    if file_name == scene_num + suffix + '.png':
                        scenes[scene_num][i] = directory + file_name

        # Only keep scenes that have exactly 5 files and no None entries
        full_scenes = {k: v for k, v in scenes.items() if len(v) == 5 and all(v)}

        # Convert the dictionary values to a list
        full_scenes_list = list(full_scenes.values())

        # Randomly shuffle the list
        random.shuffle(full_scenes_list)

        # Split the list into a training set (80%) and a test set (20%)
        train_size = int(len(full_scenes_list) * 0.8)
        train_scenes = full_scenes_list[:train_size]
        test_scenes = full_scenes_list[train_size:]

        # Write the training set to a text file
        with open('structured3d_train.txt', 'w') as f:
            for scene in train_scenes:
                f.write('\t'.join(scene) + '\n')

        # Write the test set to a text file
        with open('structured3d_test.txt', 'w') as f:
            for scene in test_scenes:
                f.write('\t'.join(scene) + '\n')

    if write_tf_record:
        # write to TFRecord
        train_paths = open(train_file, 'r').read().splitlines()
        test_paths = open(test_file, 'r').read().splitlines()
        for i in range(len(train_paths)):
            current = train_paths[i]
            paths = current.split('\t')
            print("paths ", paths[0]) #.jpg
            print(paths[1]) # wall
            print(paths[2]) # close
            print(paths[3]) # rooms
            print(paths[4]) # close_wall
            break
        

        write_bd_rm_record(train_paths, name='tf2deep_train.tfrecords')
        write_bd_rm_record(test_paths, name='tf2deep_test.tfrecords')