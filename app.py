import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time

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
# HEADER
# ============================================================

st.title("🔬 LRDRN Image Restoration")

st.markdown(
    """
    ### KLA Grayscale Image Super-Resolution

    Restore low-resolution grayscale images from **128 × 128**
    to **256 × 256** using a trained **Lightweight Residual
    Dense Restoration Network (LRDRN)**.
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

    # Support both checkpoint formats:
    # 1. {"model_state_dict": ...}
    # 2. direct state_dict

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


try:

    model = load_model()

    st.success(
        "✅ LRDRN model is ready for image restoration"
    )

except Exception as e:

    st.error(
        "❌ Could not load the LRDRN model."
    )

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


# ============================================================
# SIDEBAR — MODEL PERFORMANCE
# ============================================================

st.sidebar.header("🏆 Model Performance")

col1, col2 = st.sidebar.columns(2)

with col1:

    st.metric(
        "PSNR",
        "26.7967 dB"
    )

with col2:

    st.metric(
        "SSIM",
        "0.6811"
    )

st.sidebar.metric(
    "Super-Resolution Scale",
    "2×"
)

st.sidebar.metric(
    "Parameters",
    "751,873"
)

st.sidebar.caption(
    "PSNR and SSIM are benchmark results obtained during model evaluation."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a grayscale PNG or JPG image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# IMAGE PROCESSING
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
        # RESIZE INPUT TO 128 × 128
        # ----------------------------------------------------

        image_128 = image.resize(
            (128, 128),
            Image.Resampling.BICUBIC
        )


        # ----------------------------------------------------
        # CONVERT IMAGE TO NUMPY
        # ----------------------------------------------------

        input_array = (
            np.asarray(
                image_128,
                dtype=np.float32
            ) / 255.0
        )


        # ----------------------------------------------------
        # CONVERT TO PYTORCH TENSOR
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(
            input_array
        ).unsqueeze(0).unsqueeze(0)

        input_tensor = input_tensor.to(device)


        # ----------------------------------------------------
        # LRDRN RESTORATION
        # ----------------------------------------------------

        start_time = time.perf_counter()

        with torch.no_grad():

            restored = model(
                input_tensor
            )

            restored = torch.clamp(
                restored,
                0,
                1
            )

        end_time = time.perf_counter()


        # ----------------------------------------------------
        # CALCULATE PROCESSING TIME
        # ----------------------------------------------------

        processing_time = (
            end_time - start_time
        )


        # ----------------------------------------------------
        # CONVERT LRDRN OUTPUT TO NUMPY
        # ----------------------------------------------------

        output_array = (
            restored[0, 0]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # BICUBIC BASELINE
        # ----------------------------------------------------

        bicubic_array = np.asarray(
            image_128.resize(
                (256, 256),
                Image.Resampling.BICUBIC
            ),
            dtype=np.float32
        ) / 255.0


        # ====================================================
        # IMAGE COMPARISON
        # ====================================================

        st.header("🖼️ Image Comparison")

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with col1:

            st.subheader("Input")

            st.image(
                input_array,
                caption="128 × 128 Grayscale",
                clamp=True,
                use_container_width=True
            )


        # ----------------------------------------------------
        # BICUBIC
        # ----------------------------------------------------

        with col2:

            st.subheader("Bicubic")

            st.image(
                bicubic_array,
                caption="256 × 256 Bicubic",
                clamp=True,
                use_container_width=True
            )


        # ----------------------------------------------------
        # LRDRN
        # ----------------------------------------------------

        with col3:

            st.subheader("LRDRN")

            st.image(
                output_array,
                caption="256 × 256 Restored",
                clamp=True,
                use_container_width=True
            )


        # ====================================================
        # OUTPUT INFORMATION
        # ====================================================

        st.header("📋 Output Information")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Height",
                output_array.shape[0]
            )

        with c2:

            st.metric(
                "Width",
                output_array.shape[1]
            )

        with c3:

            st.metric(
                "Pixel Range",
                f"{output_array.min():.3f} – "
                f"{output_array.max():.3f}"
            )


        # ====================================================
        # PROCESSING INFORMATION
        # ====================================================

        st.subheader("⚙️ Processing Information")

        p1, p2, p3 = st.columns(3)

        with p1:

            st.metric(
                "Input Resolution",
                "128 × 128"
            )

        with p2:

            st.metric(
                "Output Resolution",
                "256 × 256"
            )

        with p3:

            st.metric(
                "LRDRN Processing Time",
                f"{processing_time:.3f} sec"
            )


        # ====================================================
        # DOWNLOAD RESTORED IMAGE
        # ====================================================

        st.header("⬇️ Download Result")

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


        # ====================================================
        # PERFORMANCE NOTE
        # ====================================================

        st.info(
            "PSNR and SSIM shown in the sidebar are benchmark "
            "results obtained during model evaluation. "
            "They are not calculated for the uploaded image "
            "because a matching ground-truth image was not provided."
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Error while processing the uploaded image."
        )

        st.exception(e)


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.header("📚 About the Project")

with st.expander("🧠 About LRDRN"):

    st.write(
        """
        LRDRN (Lightweight Residual Dense Restoration Network)
        is used in this project for grayscale image
        super-resolution.

        The network receives a 128 × 128 grayscale image
        and produces a 256 × 256 restored image.
        """
    )


with st.expander("📊 Model Performance"):

    st.write(
        """
        **Improved LRDRN benchmark results**

        PSNR: **26.7967 dB**

        SSIM: **0.6811**

        Best Epoch: **38**
        """
    )


with st.expander("📁 Dataset"):

    st.write(
        """
        Training images: **2,560**

        Validation images: **640**

        Test images: **400**

        Image type: **Grayscale**
        """
    )


with st.expander("⚙️ Methodology"):

    st.write(
        """
        1. Upload a grayscale image.

        2. The image is converted to grayscale.

        3. The input is resized to 128 × 128.

        4. The trained LRDRN model performs restoration.

        5. The output is generated at 256 × 256.

        6. Bicubic interpolation is also generated as a
           baseline for visual comparison.

        7. The LRDRN restored image can be downloaded as PNG.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LRDRN Image Restoration Dashboard | KLA Dataset"
)
