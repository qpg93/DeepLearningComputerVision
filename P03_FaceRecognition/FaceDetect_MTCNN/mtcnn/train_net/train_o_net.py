import os
import sys
REPO_DIR = os.getcwd()
MTCNN_DIR = os.path.join(REPO_DIR, 'P03_FaceRecognition', 'FaceDetect_MTCNN', 'mtcnn')
MTCNN_CORE_DIR = os.path.join(MTCNN_DIR, 'core')
MTCNN_TRAIN_NET_DIR = os.path.join(MTCNN_DIR, 'train_net')
sys.path.append(MTCNN_DIR)
sys.path.append(MTCNN_CORE_DIR)
sys.path.append(MTCNN_TRAIN_NET_DIR)
# mtcnn.core
from imagedb import ImageDB
# mtcnn.train_net
import train
# mtcnn
import config



def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=True):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train.train_onet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr, use_cuda=use_cuda)

if __name__ == '__main__':

    print('train ONet argument:')

    ROOT_DIR = os.path.dirname(MTCNN_DIR)
    ANNO_DIR = os.path.join(ROOT_DIR, 'anno_store')
    annotation_file = os.path.join(ANNO_DIR, "imglist_anno_48.txt")
    model_store_path = os.path.join(ROOT_DIR, "model_store")
    end_epoch = 30
    lr = 0.001
    batch_size = 64

    use_cuda = True
    frequent = 50


    train_net(annotation_file, model_store_path,
                end_epoch, frequent, lr, batch_size, use_cuda)
