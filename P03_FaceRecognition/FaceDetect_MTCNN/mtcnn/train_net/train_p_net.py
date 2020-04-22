import argparse
import sys
import os
REPO_DIR = os.getcwd()
MTCNN_DIR = os.path.join(REPO_DIR, 'P03_FaceRecognition', 'FaceDetect_MTCNN', 'mtcnn')
MTCNN_CORE_DIR = os.path.join(MTCNN_DIR, 'core')
MTCNN_DATAPREP_DIR = os.path.join(MTCNN_DIR, 'data_preprocess')
MTCNN_TRAIN_NET_DIR = os.path.join(MTCNN_DIR, 'train_net')
sys.path.append(MTCNN_DIR)
sys.path.append(MTCNN_CORE_DIR)
sys.path.append(MTCNN_DATAPREP_DIR)
sys.path.append(MTCNN_TRAIN_NET_DIR)
# mtcnn.core
from imagedb import ImageDB
# mtcnn.train_net
from train import train_pnet
# mtcnn.config
import config as config

ROOT_DIR = os.path.dirname(MTCNN_DIR)

annotation_file = os.path.join(ROOT_DIR, 'anno_store', 'imglist_anno_12.txt')
model_store_path = os.path.join(ROOT_DIR, 'model_store')
end_epoch = 30
frequent = 200
lr = 0.01
batch_size = 512
use_cuda = True


def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=False):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    train_pnet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr, use_cuda=use_cuda)

def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--anno_file', dest='annotation_file',
                        default=os.path.join(config.ANNO_STORE_DIR,config.PNET_TRAIN_IMGLIST_FILENAME), help='training data annotation file', type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODEL_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='', help='training data annotation images prefix root path', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # args = parse_args()
    print('train Pnet argument:')
    # print(args)

    train_net(annotation_file, model_store_path,
                end_epoch, frequent, lr, batch_size, use_cuda)

    # train_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
    #             end_epoch=args.end_epoch, frequent=args.frequent, lr=args.lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
