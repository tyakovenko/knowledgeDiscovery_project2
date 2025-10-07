import os
from PIL import Image
import numpy as np
import pandas as pd
#0 label for fogsmog
#1 label for rime


def process_jpg_images_to_dataframe(folder_path, num):
    # Initialize an empty list to store the data records
    data_records = []
    allImgs = []
    allModes = []

    # Define the single extension we are looking for
    JPG_EXTENSION = '.jpg'

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Create the full file path
        img_path = os.path.join(folder_path, filename)

        # Check if the file is a regular file and ends with .jpg (case-insensitive)
        if os.path.isfile(img_path) and filename.lower().endswith(JPG_EXTENSION):
            try:
                print(f"--- Processing: {filename} ---")

                # Load the image
                img = Image.open(img_path)

                # Convert to NumPy array
                img_array = np.array(img)
                allImgs.append(img_array)


                image_mode = img.mode
                allModes.append(image_mode)

                # Get the total number of elements in the array (H * W * C)
                image_size = img_array.size


                # Append the filename and the NumPy array to the list
                data_records.append({
                    'filename': 'r' + filename,
                    'img_array': img_array,
                    'label': num, # 0 for fogsmog
                    'size': image_size,
                    'shape': img_array.shape,
                    'img_dimensions': img_array.ndim,
                    'mode':image_mode,
                })

                # Close the image file
                img.close()

            except Exception as e:
                print(f"Could not process {filename}: {e}")

    # Create the Pandas DataFrame from the list of records
    #df = pd.DataFrame(data_records)
    allLables = np.ones(len(data_records))
    dfSmall = pd.DataFrame({
        'imgArray': allImgs,
        'label': allLables,
        'mode': allModes,
    })

    return allImgs, allLables, allModes



def combineDfs (df1, df2):
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df


# --- Set the folder path (You must update this line!) ---
#image_folder = "/home/taya/PycharmProjects/knowledgeDiscovery_project2/rime"
#allImgs, allLables, allModes = process_jpg_images_to_dataframe(image_folder,0)
#image_df.to_csv("rime_data.csv", index=False)
#dfSmall.to_csv("rime_data_small.csv", index=False)
def combineAll ():
    fogDf = pd.read_csv("intemediate/fogsmog_data_small.csv")
    rimeDf = pd.read_csv("intemediate/rime_data_small.csv")

    combined_df = combineDfs(fogDf, rimeDf)
    combined_df.to_csv("combined_data_small.csv", index=False)
# Run the function and create the DataFrame
combineAll()
# Display the first few rows of the DataFrame
#print("\n--- DataFrame Head ---")
#print(image_df.head())

#combineAll()