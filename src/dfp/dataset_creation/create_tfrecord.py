import os
import random
from tf_record import *

train_file = 'structured3d_activities_train.txt'
test_file = 'structured3d_activities_test.txt'

# debug
if __name__ == '__main__':
    write_new_txt = True
    write_tf_record = True
    include_activities = True

    if write_new_txt:

        # write the r3d_train.txt and r3d_test.txt
        # Specify your directory
        directory = '/home/amohap/MT/code/t-mt-2023-FloorplanReconstruction-AdrianaMohap/source/trajectory_sampling/plots/tf2deep_act/'

        # List all files in directory
        all_files = os.listdir(directory)

        # Initialize dictionary to hold all scenes
        scenes = {}

        # Order of the files for a scene
        if include_activities:
            order = ['', '_wall', '_close', '_rooms', '_close_wall', '_act_opening_door', '_act_sitting', '_act_laying', '_act_washing_hands']
        else:
            order = ['', '_wall', '_close', '_rooms', '_close_wall']

        # Go through each file
        for file_name in all_files:
            if file_name.endswith('.png'):
                # Extract the scene number (first 5 characters of the file name)
                scene_num = file_name[:5]

                # If the scene number is not in the dictionary yet, add an empty list for it
                if scene_num not in scenes:
                    if include_activities:
                        scenes[scene_num] = [None]*9
                    else:
                        scenes[scene_num] = [None]*5

                # Add the file name to the correct position in the scene's list in the dictionary
                for i, suffix in enumerate(order):
                    if file_name == scene_num + suffix + '.png':
                        scenes[scene_num][i] = directory + file_name

        # Only keep scenes that have exactly 5 files and no None entries
        if include_activities:
            full_scenes = {k: v for k, v in scenes.items() if len(v) == 9 and all(v)}
        else:
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
        with open(train_file, 'w') as f:
            for scene in train_scenes:
                f.write('\t'.join(scene) + '\n')

        # Write the test set to a text file
        with open(test_file, 'w') as f:
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
            
            if include_activities:
                print(paths[5]) # act_door
                print(paths[6]) # act_sitt
                print(paths[7]) # act_laying
                print(paths[8]) # act_washing
            
            break
        
        if include_activities:
            write_bd_rm_act_record(train_paths, name='tf2deep_act_train.tfrecords')
            write_bd_rm_act_record(test_paths, name='tf2deep_act_test.tfrecords')
        else:
            write_bd_rm_record(train_paths, name='tf2deep_train.tfrecords')
            write_bd_rm_record(test_paths, name='tf2deep_test.tfrecords')