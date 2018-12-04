from PIL import Image
from resizeimage import resizeimage
import os

reject,take = 0,0
for root,dir,files in os.walk('Images/'):
    print(root)
    folder = f'train/{root[10:]}'
    if len(root) > 12 and not os.path.exists(folder):
        os.mkdir(folder)
    for file in files:
        with open(f'{root}/{file}', 'rb') as f:
            with Image.open(f) as image:
                try:
                    cover = resizeimage.resize_cover(image,[300,300])
                    cover.save(f'{folder}/{file}', image.format)
                    take+=1
                except:
                    reject+=1
    print (reject,take)
