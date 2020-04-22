import os
import sys
sys.path.append(os.getcwd())
import assemble

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
ANNO_DIR = os.path.join(ROOT_DIR, 'anno_store')
print(ANNO_DIR)

rnet_postive_file = os.path.join(ANNO_DIR, 'pos_24.txt')
rnet_neg_file = os.path.join(ANNO_DIR, 'neg_24.txt')
rnet_part_file = os.path.join(ANNO_DIR, 'part_24.txt')
rnet_landmark_file = os.path.join(ANNO_DIR, 'landmark_24.txt')
imglist_filename = os.path.join(ANNO_DIR, 'imglist_anno_24.txt')

if __name__ == '__main__':

    anno_list = []

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
