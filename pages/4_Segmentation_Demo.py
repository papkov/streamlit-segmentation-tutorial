import streamlit as st
from streamlit.hello.utils import show_code
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image

import segmentation_models_pytorch as smp


def segmentation_demo():
    model_weights = st.file_uploader("Choose a model file")
    image = st.file_uploader("Choose an image")

    if image is not None:
        image = Image.open(image).convert("RGB")
        st.image(image, "Uploaded image")

    if model_weights is not None:
        model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model.load_state_dict(torch.load(model_weights,
                                         map_location="cpu"))
        model.eval()


        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225)),
        ])

        if image is not None:
            tensor = transforms(tv_tensors.Image(image))[None, ...]
            with torch.no_grad():
                mask = model(tensor).numpy().clip(0, 1)
            st.image(mask[0, 0], "Predicted mask")


st.set_page_config(page_title="Segmentation Demo", page_icon="🔬")
st.markdown("# Segmentation Demo")
st.sidebar.header("Segmentation Demo")


segmentation_demo()

show_code(segmentation_demo)