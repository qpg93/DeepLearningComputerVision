import os
import sys
import assemble

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
ANNO_DIR = os.path.join(ROOT_DIR, 'anno_store')
print(ANNO_DIR)

onet_postive_file = os.path.join(ANNO_DIR, 'pos_48.txt')
onet_neg_file = os.path.join(ANNO_DIR, 'neg_48.txt')
onet_part_file = os.path.join(ANNO_DIR, 'part_48.txt')
onet_landmark_file = os.path.join(ANNO_DIR, 'landmark_48.txt')
imglist_filename = os.path.join(ANNO_DIR, 'imglist_anno_48.txt')

if __name__ == '__main__':

    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename)
