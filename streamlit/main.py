import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

def show_anns(anns:list) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # color_mask = np.concatenate([[0.9, 0.99, 0.1], [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def filter_masks(masks:list)->list:
    areas = [mask['area'] for mask in masks]
    max_areai = areas.index(max(areas))
    background = masks[max_areai]['segmentation']
    new_masks = []
    for mask in masks:
        m1 = mask['segmentation']
        inter = np.logical_and(background, m1)
        inter_per = np.count_nonzero(inter)/np.count_nonzero(m1)
        if inter_per <= 0.5:
            new_masks.append(mask)          
    return new_masks

def plot_seg(image:np.ndarray, masks:list) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image)
    show_anns(masks)
    ax.axis('off')
    fig.tight_layout(pad=0)
    st.pyplot(fig)

### MAIN APP ###
st.title("Segment image with SAM")

with st.sidebar:
    st.sidebar.title("SAM parameters")
    with st.form(key="img_load"):
        points_per_side = st.slider('Points per side', 1, 128, 32)
        pred_iou_thresh = st.slider('Pred IOU threshold', 0.5, 1.0, 0.86)

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","tif"])
        submit = st.form_submit_button("Load image and model parameters")
    
if submit:
    if uploaded_file is None:
        st.text('Upload some image first')
    else:
        st.session_state['points_per_side'] = points_per_side
        st.session_state['pred_iou_thresh'] = pred_iou_thresh
        st.session_state['file'] = uploaded_file
        st.session_state['segmented'] = False
        st.session_state['submitted'] = True

        image = cv2.imdecode(np.frombuffer(st.session_state['file'].read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.session_state['image'] = image

if 'submitted' in st.session_state:
    st.subheader('Uploaded image:')
    st.image(st.session_state['image'], use_column_width=True)
    
    with st.form(key="search_pept"):
        search_submit = st.form_submit_button("Segment image")

    with st.spinner('Calculating masks'):
        if search_submit:
            mask_generator = SamAutomaticMaskGenerator(sam, 
                                points_per_side=st.session_state['points_per_side'],
                                pred_iou_thresh=st.session_state['pred_iou_thresh'])
            st.session_state['masks'] = mask_generator.generate(st.session_state['image'])
            st.subheader('Image segmented with SAM:')
            plot_seg(st.session_state['image'], st.session_state['masks'])
            st.session_state['nm'] = filter_masks(st.session_state['masks'])
            st.subheader('Segmentation masks filtered:')
            plot_seg(st.session_state['image'], st.session_state['nm'])
