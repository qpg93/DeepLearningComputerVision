## Data Augmentation of Images
Here are the scripts for generating images in the purpose of data augmentation. The script is based on traditional computer vision techniques, including the functions below:  
* Image crop
* Color shift
* Rotation transform
* Perspective transform

### 1. DataAugment_Auto.py
__DataAugment_Auto.py__ is the first script created, which can automatically and randomly apply all 4 functions to the target image and generate and save 20 images in the same directory.  
1. Copy and paste the target image in the same directory of the script __DataAugment_Auto.py__.
2. Make sure that the image's name is __image.jpg__.
3. Execute the script __DataAugment_Auto.py__.
4. Check the 20 generated images whose names are __image_1.jpg__ to __image_20.jpg__ in the same directory.

### 2. DataAugment.py
__DataAugment.py__ is the script more adaped to industrial use. All images in the input directory will be processed and saved to the output directory.
#### 2.1 User guide
```bash
python DataAugment.py -h
```
#### 2.2 Examples
* Image crop
```bash
python DataAugment.py --img_input "image_input_dir" --img_output "image_output_dir" --scale  0.7
```
* Color shift
```bash
python DataAugment.py --img_input "image_input_dir" --img_output "image_output_dir" --shift 60
```
* Rotation transform
```bash
python DataAugment.py --img_input "image_input_dir" --img_output "image_output_dir" --angle 30 --scale 1.5
```
* Perspective transform
```bash
python DataAugment.py --img_input "image_input_dir" --img_output "image_output_dir" --margin 50
```