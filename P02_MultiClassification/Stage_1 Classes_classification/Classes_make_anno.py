import pandas as pd
import os
from PIL import Image


CURRENT_DIR = os.path.dirname(__file__)
ROOTS = os.path.join(os.path.dirname(CURRENT_DIR), 'Dataset')
PHASE = ['train', 'val']
CLASSES = ['Mammals', 'Birds']  # [0,1]
SPECIES = ['rabbits', 'chickens']


DATA_info = {'train': {'path': [], 'classes': []},
             'val': {'path': [], 'classes': []}
             }
for p in PHASE:
    for s in SPECIES:
        DATA_DIR = os.path.join(ROOTS, p, s,)
        print(DATA_DIR)
        DATA_NAME = os.listdir(DATA_DIR)

        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))
            except OSError:
                pass
            else:
                DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))
                if s == 'rabbits':
                    DATA_info[p]['classes'].append(0)
                else:
                    DATA_info[p]['classes'].append(1)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv(os.path.join(CURRENT_DIR, 'Classes_%s_annotation.csv' % p))
    print('Classes_%s_annotation file is saved.' % p)
