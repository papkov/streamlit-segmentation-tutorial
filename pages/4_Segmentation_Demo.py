import streamlit as st
from streamlit.hello.utils import show_code

import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image

import segmentation_models_pytorch as smp


def segmentation_demo():
    model_weights = st.file_uploader("Choose a model file")
    image = st.file_uploader("Choose an image")

    if model_weights is not None and image is not None:
        image = Image.open(self.images[i]).convert("RGB")
        st.image(image, "Uploaded image")

        model = torch.load_state_dict(model_weights)
        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )


        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

        mask = model(transforms(image)).numpy()
        st.image(mask[0], "Uploaded image")


st.set_page_config(page_title="Segmentation Demo", page_icon="🔬")
st.markdown("# Segmentation Demo")
st.sidebar.header("Segmentation Demo")


segmentation_demo()

show_code(segmentation_demo)