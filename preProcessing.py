import numpy as np
import cv2
from typing import List
import pandas as pd
import process as p

'''
df = pd.read_csv('combined_data_small.csv')
fogImgs, fogLabels, fogMode = p.process_jpg_images_to_dataframe("/home/taya/PycharmProjects/knowledgeDiscovery_project2/fogsmog", 0)
rimeImgs, rimeLabels, rimeMode = p.process_jpg_images_to_dataframe("/home/taya/PycharmProjects/knowledgeDiscovery_project2/rime", 1)

#only RGB since those are the only ones with dim 3
fogIdKeep =  [i for i, arr in enumerate(fogImgs) if arr.ndim == 3]
rimeIdKeep = [i for i, arr in enumerate(rimeImgs) if arr.ndim == 3]

#filter all the arrays based on indeces to keep; do not need mode
fogImgs_filtered = [fogImgs[i] for i in fogIdKeep]
fogLabels_filtered = [fogLabels[i] for i in fogIdKeep]
rimeImgs_filtered = [rimeImgs[i] for i in rimeIdKeep]
rimelabels_filtered = [rimeLabels[i] for i in rimeIdKeep]


#invert and turn into balck and white each  image in the list of images
def process_images(image_list: List[np.ndarray]) -> List[np.ndarray]:
    processed_list = []

    for image_array in image_list:
        # 1. Ensure the data type is 8-bit unsigned integer (0-255 range)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # 2. Invert the image ↩️
        # Subtract every pixel value from 255
        inverted_image = 255 - image_array

        # 3. Convert to Grayscale
        # If the image has 3 color channels (e.g., RGB/BGR), convert it.
        if len(inverted_image.shape) == 3:
            grayscale_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
        else:
            # If it's already 1D (flattened), 2D (grayscale), or 1-channel
            # we can use it directly as grayscale.
            grayscale_image = inverted_image

        # 4. Convert to Black and White (Binarization) ⚫⚪
        # cv2.threshold uses a threshold (127) to make every pixel either 0 or 255.
        # The first returned value is the threshold used, which we ignore (_).
        _, binary_image = cv2.threshold(
            grayscale_image,
            127,  # Threshold value (e.g., 127)
            255,  # Max value for pixels above the threshold
            cv2.THRESH_BINARY
        )

        processed_list.append(binary_image)

    return processed_list

processedFog = process_images(fogImgs_filtered)
processedRime = process_images(rimeImgs_filtered)

def getListSizes (imgs):
    sizes = []
    for img in imgs:
        sizes.append(img.size)

    return sizes

#dfs for both saved into csvs
fog = {
    'img': processedFog,
    'sizes': getListSizes(processedFog),
    'labels': fogLabels_filtered,
}
rime = {
    'img': processedRime,
    'sizes': getListSizes(processedRime),
    'labels': rimelabels_filtered,
}

dfFog = pd.DataFrame(fog)
dfRime = pd.DataFrame(rime)

dfFog.to_csv('fogProcessed.csv', index=False)
dfRime.to_csv('rimeProcessed.csv', index=False)
'''

#combine dfs
dfFog = pd.read_csv('intemediate/fogProcessed.csv')
dfRime = pd.read_csv('intemediate/rimeProcessed.csv')

processedCombo = p.combineDfs(dfFog, dfRime)
#find percentiles and drop images with sizes in the 10th and 90th percentile
arrSize = np.array(processedCombo['sizes'])
p10Size = np.percentile(arrSize, 10)
p90Size = np.percentile(arrSize, 90)

filter = (processedCombo['sizes'] > p10Size) & (processedCombo['sizes'] < p90Size)
finalProcess = processedCombo[filter].copy()
finalProcess.to_csv('finalProcessed.csv', index=False)

#some class inbalance is still present and preserved


