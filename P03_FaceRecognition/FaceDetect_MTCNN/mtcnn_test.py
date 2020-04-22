import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face

CURRENT_DIR = os.path.dirname(__file__)

PNET_PATH = os.path.join(CURRENT_DIR, 'model_for_use', 'pnet_epoch.pt')
RNET_PATH = os.path.join(CURRENT_DIR, 'model_for_use', 'rnet_epoch.pt')
ONET_PATH = os.path.join(CURRENT_DIR, 'model_for_use', 'onet_epoch.pt')

if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path=PNET_PATH,
                                        r_model_path=RNET_PATH,
                                        o_model_path=ONET_PATH,
                                        use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread(CURRENT_DIR + '/eu1_input.jpg')
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align
    save_name = CURRENT_DIR + '/eu1_output.png'
    vis_face(img_bg,bboxs,landmarks, save_name)