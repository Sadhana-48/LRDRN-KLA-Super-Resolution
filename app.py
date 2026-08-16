import streamlit as st
import torch
import numpy as np
from PIL import Image
import io

from model import LRDRN


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LRDRN Image Restoration",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔬 LRDRN Image Restoration")

st.markdown(
    """
    **Lightweight Residual Dense Restoration Network (LRDRN)**

    Restore a low-resolution grayscale image from **128 × 128**
    to **256 × 256** using the trained LRDRN model.
    """
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


try:

    model = load_model()

    st.success(
        "✅ Trained LRDRN model loaded successfully"
    )

except Exception as e:

    st.error("❌ Model could not be loaded")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR — MODEL INFORMATION
# ============================================================

st.sidebar.header("📊 Model Information")

st.sidebar.write("**Model:** LRDRN")
st.sidebar.write("**Parameters:** 751,873")
st.sidebar.write("**Best Epoch:** 38")
st.sidebar.write("**Input:** 128 × 128")
st.sidebar.write("**Output:** 256 × 256")
st.sidebar.write("**Type:** Grayscale")
st.sidebar.write(f"**Device:** {device}")

st.sidebar.divider()

st.sidebar.header("🏆 Model Benchmark")

st.sidebar.metric(
    "PSNR",
    "26.7967 dB"
)

st.sidebar.metric(
    "SSIM",
    "0.6811"
)

st.sidebar.caption(
    "Benchmark values obtained during model evaluation."
)


# ============================================================
# UPLOAD
# ============================================================

st.header("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a grayscale PNG or JPG image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# RESTORATION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("L")

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image_128 = image.resize(
            (128, 128),
            Image.Resampling.BICUBIC
        )

        input_array = (
            np.asarray(
                image_128,
                dtype=np.float32
            ) / 255.0
        )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(
            input_array
        ).unsqueeze(0).unsqueeze(0)

        input_tensor = input_tensor.to(device)

        # ----------------------------------------------------
        # RESTORE
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

        output_array = (
            restored[0, 0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        st.header("🖼️ Restoration Result")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Input")

            st.image(
                input_array,
                caption="128 × 128 Grayscale",
                clamp=True,
                use_container_width=True
            )

        with col2:

            st.subheader("LRDRN Restored")

            st.image(
                output_array,
                caption="256 × 256 Restored Output",
                clamp=True,
                use_container_width=True
            )

        # ----------------------------------------------------
        # OUTPUT INFORMATION
        # ----------------------------------------------------

        st.header("📋 Output Information")

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
            "Pixel Range",
            f"{output_array.min():.3f} – "
            f"{output_array.max():.3f}"
        )

        # ----------------------------------------------------
        # DOWNLOAD
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
        # IMPORTANT NOTE
        # ----------------------------------------------------

        st.info(
            "PSNR and SSIM shown in the sidebar are the "
            "model's benchmark results. They are not calculated "
            "for the uploaded image because a matching ground-truth "
            "image was not provided."
        )

    except Exception as e:

        st.error(
            "❌ Error while processing the image."
        )

        st.exception(e)


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.header("📚 About the Project")

st.markdown(
    """
    This project uses a **Lightweight Residual Dense Restoration
    Network (LRDRN)** for grayscale image super-resolution.

    **Dataset**
    - Training samples: 2,560
    - Validation samples: 640
    - Test images: 400

    **Model**
    - Parameters: 751,873
    - Best epoch: 38
    - Input resolution: 128 × 128
    - Output resolution: 256 × 256

    **Evaluation**
    - PSNR: 26.7967 dB
    - SSIM: 0.6811
    """
)
