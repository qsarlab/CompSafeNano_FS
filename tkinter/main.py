import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#SAM params
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def open_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")])
    if filepath:
        global image_path
        image_path = filepath
        image = Image.open(filepath)
        wid, hei = image.size
        global rx, ry
        rx, ry = wid/600, hei/500
        image = image.resize((600, 500))  # Resize the image to fit the window
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo
        ### For SAM model and plt displaying
        global image_sam
        image_sam = cv2.imread(image_path)
        image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_sam)

def save_masks(name:str, i:int, mask:np.array)-> None:
    # Convert the boolean array to an integer array
    integer_array = mask.astype(np.uint8)
    os.makedirs('img/cut', exist_ok=True)
    # # Multiply the integer array by a value (e.g., 255)
    # binary_array = integer_array * 255
    # # Convert the integer array to an image and save it
    # binary_image = Image.fromarray(binary_array)
    # binary_image.save(f"img/whole/{name}_{i}_all.tif")
    
    # Find the indices of non-zero elements
    indices = np.nonzero(integer_array)
    # Determine the bounds and extract the non-zero region
    min_x, min_y = np.min(indices, axis=1)
    max_x, max_y = np.max(indices, axis=1)
    non_zero_region = integer_array[min_x:max_x, min_y:max_y]
    pad_width = 10  # Width of the padding
    padded_array = np.pad(non_zero_region, pad_width, mode='constant')
    binary_array = padded_array * 255
    binary_image = Image.fromarray(binary_array)
    binary_image = binary_image.resize((64, 64))
    # Check if the file already exists
    cut_name = f"img/cut/{name}_{i}.tif"
    file_exists = os.path.isfile(cut_name)
    while file_exists:
        # Generate a random number
        random_number = random.randint(1, 1000)
        # Construct the new filename with the random number
        cut_name = f"img/cut/{name}_{i}_{random_number}.tif"
        # Check if the new filename already exists
        file_exists = os.path.isfile(cut_name)
    binary_image.save(cut_name)

def segment_save_mask(event):
    x = event.x
    y = event.y
    # print(f"Clicked coordinates: ({x*rx}, {y*ry})")
    input_point = np.array([[x*rx, y*ry]])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True)
    # score_max = scores.argmax()
    masks = masks[:2]
    scores = scores[:2]
    for i, (mask, score) in enumerate(zip(masks, scores)):
        seg_name = image_path.split('/')[-1].split('.')[0]
        save_masks(seg_name, i, mask)
        # plt.figure(figsize=(10,10))
        # plt.imshow(image_sam)
        # show_mask(mask, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        # plt.axis('off')
        # plt.show()

# Create the main Tkinter window
window = tk.Tk()
window.title("Image Uploader")
height = 700
width = 900
y = (window.winfo_screenheight()//2)-(height//2)
x = (window.winfo_screenwidth()//2)-(width//2)
window.geometry(f'{width}x{height}+{x}+{y}')

# Create a button to open the file dialog
button = tk.Button(window, text="Upload Image", command=open_image)
button.pack(pady=10)

# Create a label to display the uploaded image
label = tk.Label(window)
label.pack()

# Bind the left mouse click event to the label
label.bind("<Button-1>", segment_save_mask)

# Run the Tkinter event loop
window.mainloop()