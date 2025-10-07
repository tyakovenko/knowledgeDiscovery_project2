#explore the images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#all the graphics were made using PyCharm IDE

df = pd.read_csv("fullDfs/combined_data.csv")
print("The number of images in the dataset is ", len(df))
print("The number of images in each class is ", df['label'].value_counts())
print("The image modes represented in the dataframe are the following: ", df['mode'].value_counts())

def classCountHistogram (df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of labels
    df['label'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Number of Images per Class')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Histogram of image modes
    df['mode'].value_counts().plot(kind='bar', ax=axes[1], color='salmon', edgecolor='black')
    axes[1].set_title('Image Modes Distribution')
    axes[1].set_xlabel('Image Mode')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # the image sizes are not consistent; thus it will need to be normalized in teh data preprocessing stage
    # below are the statistics
    dfStats = df.describe()
    dfStats.to_csv("stats.csv")

