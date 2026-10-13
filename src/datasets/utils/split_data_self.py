import os.path

import cv2
import numpy as np

from pathlib import Path
import mediapipe as mp
import time
import gc
import tracemalloc

tracemalloc.start()

def load_split_nvgesture(file_with_split='./nvgesture_train_correct.lst', list_split=list()):
    with open(file_with_split, 'rt') as f:
        dict_name = file_with_split[file_with_split.rfind('/') + 1:]
        dict_name = dict_name[:dict_name.find('_')]

        for line in f:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1
                    params_dictionary['label'] = label
                elif key in ('depth', 'color', 'duo_left'):
                    # first store path
                    params_dictionary[key] = path + '/' + parsed[1]
                    # store start frame
                    params_dictionary[key + '_start'] = int(parsed[2])

                    params_dictionary[key + '_end'] = int(parsed[3])

            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']

            list_split.append(params_dictionary)

    return list_split


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.5)
def load_data_from_file(data_path, example_config, sensor, image_width, image_height, nogesture = False):
    path = example_config[sensor] + ".avi"
    path = Path(data_path) / path[path.find('/') + 1:]
    start_frame = example_config[sensor + '_start']
    end_frame = example_config[sensor + '_end']
    label = example_config['label']

    if end_frame - start_frame > 80:
        new_start = (end_frame - start_frame) // 2 - 40 + start_frame
        new_end = (end_frame - start_frame) // 2 + 40 + start_frame
        start_frame = new_start
        end_frame = new_end

    chnum = 3 if sensor == "color" else 1

    video_container = np.zeros((image_height, image_width, chnum, 160 if nogesture else 80), dtype=np.uint8)

    cap = cv2.VideoCapture(str(path))
    #
    if nogesture:
        start_offset = 40 if start_frame >= 40 else start_frame
        end_offset = 40 if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - end_frame >= 40 \
            else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - end_frame
        frames_to_load = range(start_frame - start_offset, end_frame + end_offset)
    else:
        frames_to_load = range(start_frame, end_frame)
    #
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for indx, frameIndx in enumerate(frames_to_load):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (image_width, image_height))
            if sensor != "color":
                frame = frame[..., 0]
                frame = frame[..., np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")
    #
    #


    #
    for i in range(video_container.shape[3]):
        img=video_container[:,:,:,i]
        skelect=np.zeros([video_container.shape[3],21,2])
        frame = cv2.flip(img, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_handedness:
            for hand_label in results.multi_hand_landmarks:
                count = 0

                for mark in hand_label.landmark:
                    x,y=mark.x,mark.y
                    skelect[i,count,0]=x
                    skelect[i, count, 1] = y
                    count+=1
                break

    print(skelect.sum())
    # if not os.path.isdir(os.path.join(r'E:\datasets\NVGesture',example_config['depth'].split('/')[-3])):
    #     os.mkdir(os.path.join(r'E:\datasets\NVGesture',example_config['depth'].split('/')[-3]))
    #
    # count=len(os.listdir(os.path.join(r'E:\datasets\NVGesture',example_config['depth'].split('/')[-3])))
    #
    # np.save(os.path.join(r'E:\datasets\NVGesture',example_config['depth'].split('/')[-3],'{}.npy'.format(str(count).rjust(2,'0'))),skelect)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    #
    # print("[Top 10 memory usage]")
    # for stat in top_stats[:10]:
    #     print(stat)

    del video_container
    # del skelect
    del results
    gc.collect()
    cap.release()
    # if nogesture:
    #     return video_container, label, (start_offset, end_offset)
    # else:
    #     return video_container, label, None


if __name__ == '__main__':
    file_lists = os.path.join("D:\datasets\zip_data","nvgesture_{}_correct_cvpr2016_v2.lst".format("test"))

    data_list = list()
    load_split_nvgesture(file_with_split=str(file_lists), list_split=data_list)

    print(len(data_list))

    for i in range(len(data_list)):
        load_data_from_file(r'D:\datasets\zip_data',example_config=data_list[i],sensor='color',image_width=320, image_height=240)


