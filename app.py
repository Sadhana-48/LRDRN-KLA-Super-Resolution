import streamlit as st
import torch
import numpy as np
from PIL import Image
import io

from model import LRDRN


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LRDRN Image Restoration",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔬 LRDRN Image Restoration Dashboard")

st.write(
    "Lightweight Residual Dense Restoration Network "
    "for 2× grayscale image super-resolution."
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = LRDRN().to(device)

    checkpoint = torch.load(
        "LRDRN_improved_best.pth",
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    st.success("✅ Trained LRDRN model loaded successfully!")

except Exception as e:

    st.error("❌ Could not load the trained LRDRN model.")
    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.write("**Model:** LRDRN")
st.sidebar.write("**Parameters:** 751,873")
st.sidebar.write("**Best Epoch:** 38")
st.sidebar.write("**Input:** 128 × 128")
st.sidebar.write("**Output:** 256 × 256")
st.sidebar.write("**Image Type:** Grayscale")
st.sidebar.write("**PSNR:** 26.7967 dB")
st.sidebar.write("**SSIM:** 0.6811")
st.sidebar.write(f"**Device:** {device}")


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header("Upload Input Image")

uploaded_file = st.file_uploader(
    "Upload a grayscale image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("L")

        st.write(
            f"Original image size: "
            f"{image.size[0]} × {image.size[1]}"
        )


        # ----------------------------------------------------
        # RESIZE TO MODEL INPUT
        # ----------------------------------------------------

        image_128 = image.resize(
            (128, 128),
            Image.Resampling.BICUBIC
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        input_array = np.asarray(
            image_128,
            dtype=np.float32
        ) / 255.0


        # ----------------------------------------------------
        # CONVERT TO TORCH TENSOR
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(
            input_array
        )

        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(device)


        # ----------------------------------------------------
        # LRDRN RESTORATION
        # ----------------------------------------------------

        with torch.no_grad():

            restored = model(
                input_tensor
            )

            restored = torch.clamp(
                restored,
                0,
                1
            )


        # ----------------------------------------------------
        # CONVERT OUTPUT
        # ----------------------------------------------------

        output_array = (
            restored[0, 0]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        st.header("Restoration Result")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Input — 128 × 128"
            )

            st.image(
                input_array,
                caption="Input image",
                clamp=True,
                use_container_width=True
            )


        with col2:

            st.subheader(
                "LRDRN Output — 256 × 256"
            )

            st.image(
                output_array,
                caption="Restored image",
                clamp=True,
                use_container_width=True
            )


        # ----------------------------------------------------
        # DOWNLOAD OUTPUT
        # ----------------------------------------------------

        output_uint8 = (
            output_array * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        output_image = Image.fromarray(
            output_uint8
        )

        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Restored Image",
            data=buffer.getvalue(),
            file_name="LRDRN_restored.png",
            mime="image/png"
        )


        # ----------------------------------------------------
        # OUTPUT INFORMATION
        # ----------------------------------------------------

        st.header("Output Information")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Height",
            output_array.shape[0]
        )

        c2.metric(
            "Width",
            output_array.shape[1]
        )

        c3.metric(
            "Value Range",
            f"{output_array.min():.3f} – "
            f"{output_array.max():.3f}"
        )


    except Exception as e:

        st.error(
            "❌ Error while processing the image."
        )

        st.exception(e)
