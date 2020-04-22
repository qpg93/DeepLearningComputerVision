import argparse
import os

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    ANNO_GT_DIR = os.path.join(ROOT_DIR, 'data_set', 'face_detection', 'wider_face_split')
    #print(ANNO_GT_DIR)
    file = open(ANNO_GT_DIR + '/wider_face_train_bbx_gt.txt')
    wfile = open(ANNO_GT_DIR + '/wider_face_train_mtcnn_bbx.txt', 'w')
    
    while True:
        item = ''
        line = file.readline()
        if not line:
            break
        line = line.strip()
        item += line + ' '

        index = file.readline()
        index = index.strip()
        num = int(index)

        if num == 0:
            tmp = file.readline()
            continue

        for i in range(0, num):
            bbox = file.readline()
            bbox = bbox.strip()
            bbox = bbox.split(' ')
            bbox = list(map(float, bbox[:]))
            item += str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[0] + bbox[2]) + ' ' + str(bbox[1] + bbox[3]) + ' '

        wfile.write(item + '\n')

    file.close()
    wfile.close()