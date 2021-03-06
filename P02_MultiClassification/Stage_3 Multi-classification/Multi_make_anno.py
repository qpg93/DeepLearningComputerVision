import pandas as pd
import os
from PIL import Image

CURRENT_DIR = os.path.dirname(__file__)
ROOTS = os.path.join(os.path.dirname(CURRENT_DIR), 'Dataset')
PHASE = ['train', 'val']
CLASSES = ['Mammals', 'Birds']  # [0,1]
SPECIES = ['rabbits', 'rats', 'chickens'] # [0,1,2]

DATA_info = {'train': {'path': [], 'classes':[], 'species': []},
             'val': {'path': [], 'classes':[], 'species': []}
             }

for p in PHASE:
    for s in SPECIES:
        DATA_DIR = os.path.join(ROOTS, p, s,)
        #print(DATA_DIR)
        DATA_NAME = os.listdir(DATA_DIR)

        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))
            except OSError:
                pass
            else:
                PATH = os.path.join(DATA_DIR, item)
                PATH = PATH.replace('\\', '/')
                #print(PATH)
                DATA_info[p]['path'].append(PATH)

                if s == 'rabbits' or s == 'rats':
                    DATA_info[p]['classes'].append(0)
                else:
                    DATA_info[p]['classes'].append(1)

                if s == 'rabbits':
                    DATA_info[p]['species'].append(0)
                elif s == 'rats':
                    DATA_info[p]['species'].append(1)
                else:
                    DATA_info[p]['species'].append(2)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv(os.path.join(CURRENT_DIR, 'Multi_%s_annotation.csv' % p))
    print('Multi_%s_annotation file is saved.' % p)