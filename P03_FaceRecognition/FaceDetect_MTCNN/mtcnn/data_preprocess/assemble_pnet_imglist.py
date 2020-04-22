import os
import sys

#import mtcnn.data_preprocess.assemble as assemble
import assemble

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
ANNO_DIR = os.path.join(ROOT_DIR, 'anno_store')
print(ANNO_DIR)

pnet_postive_file = os.path.join(ANNO_DIR, 'pos_12.txt')
pnet_neg_file = os.path.join(ANNO_DIR, 'neg_12.txt')
pnet_part_file = os.path.join(ANNO_DIR, 'part_12.txt')
pnet_landmark_file = os.path.join(ANNO_DIR, 'landmark_12.txt')
imglist_filename = os.path.join(ANNO_DIR, 'imglist_anno_12.txt')

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
