import os
import argparse

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    ANNO_GT_DIR = os.path.join(ROOT_DIR, 'data_set', 'face_landmark', 'CelebA', 'Anno')
    #print(ANNO_GT_DIR)
    bbox_file = open(ANNO_GT_DIR + '/list_bbox_celeba.txt')
    landmark_file = open(ANNO_GT_DIR + '/list_landmarks_celeba.txt')
    wfile = open(ANNO_GT_DIR + '/list_face_train_mtcnn_bbx_landmark.txt', 'w')

    bbox_file.readline()
    bbox_file.readline()
    landmark_file.readline()
    landmark_file.readline()

    while True:
        item = ''
        bbox_line = bbox_file.readline()
        landmark_line = landmark_file.readline()
        if not bbox_line or not landmark_line:
            break

        bbox_line = bbox_line.strip()
        bbox = bbox_line.split(' ')
        tmp_bbox = bbox
        bbox = []
        for tt in tmp_bbox:
            if tt is '':
                continue
            else:
                bbox.append(tt)
        bbox = list(map(float, bbox[1:]))
        item += tmp_bbox[0] + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[0] + bbox[2]) + ' ' + str(bbox[1] + bbox[3]) + ' '

        landmark_line = landmark_line.strip()
        landmark = landmark_line.split(' ')

        for tt in landmark[1:]:
            if tt is '':
                continue
            else:
                item += tt + ' '

        if (False):
            print(item)
        else:
            wfile.write(item + '\n')

    landmark_file.close()
    bbox_file.close()
    wfile.close()