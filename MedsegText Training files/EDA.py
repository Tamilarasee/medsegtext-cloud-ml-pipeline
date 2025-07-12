import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import textwrap
import os
import numpy as np 
# Define output directory for saving plots
OUTPUT_DIR = "visualizations/eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plots and saves the top 10 repeated text annotations
def plot_top_descriptions(df):
    
    top_descriptions = df["text"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    top_descriptions.plot(kind="barh", color="skyblue")
    plt.xlabel("Frequency")
    plt.ylabel("Text Description")
    plt.title("Top 10 Most Common Text Annotations")
    plt.gca().invert_yaxis()
    
    plot_path = os.path.join(OUTPUT_DIR, "top_descriptions.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

def check_image_sizes(df, image_dir):
    
    image_sizes = []
    
    for img_name in df["Image"]:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image_sizes.append(img.shape)

    size_df = pd.DataFrame(image_sizes, columns=["Height", "Width"])
    size_counts = size_df.value_counts().reset_index()
    size_counts.columns = ["Height", "Width", "Count"]

    plt.figure(figsize=(8, 5))
    sns.barplot(data=size_counts, x="Width", y="Height", hue="Count", palette="coolwarm")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Distribution of Image Sizes")
    
    plot_path = os.path.join(OUTPUT_DIR, "image_sizes.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def visualize_samples(df, image_dir, mask_dir, num_samples=3):
   
    samples = df.sample(num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))  

    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(image_dir, row["Image"])
        mask_path = os.path.join(mask_dir, row["Image"]) 

        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            #print('image properties', img.dtype, img.min(), img.max())  
            #print('mask properties', mask.dtype, mask.min(), mask.max(), '\nunique mask values - ', np.unique(mask))
            #print('\nimage - ', img)
            #print('\nmask - ' , mask)

            wrapped_text = "\n".join(textwrap.wrap(row["text"], width=50))  

            # Image
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title(f"Image: {row['Image']}", fontsize=10, pad=5)
            axes[i, 0].axis("off")

            # Mask
            axes[i, 1].imshow(mask, cmap="gray")
            axes[i, 1].set_title("Mask", fontsize=10, pad=5)
            axes[i, 1].axis("off")

            # Text Annotations 
            axes[i, 2].text(0.5, 0.5, wrapped_text, ha="center", va="center", fontsize=12, wrap=True, bbox=dict(facecolor="white", alpha=0.7))
            axes[i, 2].set_title("Annotation", fontsize=10)
            axes[i, 2].axis("off") 

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "sample_visualizations.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {plot_path}")
    
def run_eda(data_folder):
    annotation_path = os.path.join(data_folder, "text_annotations.xlsx")
    image_dir = os.path.join(data_folder, "frames")  
    mask_dir = os.path.join(data_folder, "masks")   

    if not os.path.exists(annotation_path):
        print(f"Error: {annotation_path} not found!")
        return

    df = pd.read_excel(annotation_path)
    
    plot_top_descriptions(df)
    check_image_sizes(df, image_dir)
    visualize_samples(df, image_dir, mask_dir, num_samples=5)

