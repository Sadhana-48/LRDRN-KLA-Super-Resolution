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
    to **256 × 256** using a trained
    **Lightweight Residual Dense Restoration Network (LRDRN)**.
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

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Remove "module." prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            new_key = key[7:]

        else:

            new_key = key

        cleaned_state_dict[new_key] = value


    model.load_state_dict(
        cleaned_state_dict
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

metric_col1, metric_col2 = st.sidebar.columns(2)

with metric_col1:

    st.metric(
        "PSNR",
        "26.7967 dB"
    )

with metric_col2:

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
    "PSNR and SSIM are benchmark results obtained "
    "during model evaluation."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a grayscale PNG, JPG, JPEG, or NPY file",
    type=[
        "png",
        "jpg",
        "jpeg",
        "npy"
    ]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        file_name = uploaded_file.name.lower()


        # ====================================================
        # READ NPY FILE
        # ====================================================

        if file_name.endswith(".npy"):

            st.info(
                "📁 NumPy array detected. Reading NPY file..."
            )

            # ------------------------------------------------
            # Load NPY
            # ------------------------------------------------

            raw_array = np.load(
                uploaded_file
            )

            # ------------------------------------------------
            # Remove unnecessary dimensions
            # ------------------------------------------------

            raw_array = np.squeeze(
                raw_array
            )

            # ------------------------------------------------
            # Check dimensions
            # ------------------------------------------------

            if raw_array.ndim != 2:

                st.error(
                    f"❌ Expected a 2D grayscale array, "
                    f"but received shape {raw_array.shape}."
                )

                st.stop()


            # ------------------------------------------------
            # Convert to float32
            # ------------------------------------------------

            input_array = raw_array.astype(
                np.float32
            )


            # ------------------------------------------------
            # Get original NPY information
            # ------------------------------------------------

            original_shape = input_array.shape

            original_min = float(
                input_array.min()
            )

            original_max = float(
                input_array.max()
            )


            # ------------------------------------------------
            # Normalize NPY values
            # ------------------------------------------------

            if original_max > 1.0:

                if original_max <= 255.0:

                    input_array = (
                        input_array / 255.0
                    )

                else:

                    min_value = input_array.min()
                    max_value = input_array.max()

                    if max_value > min_value:

                        input_array = (
                            input_array - min_value
                        ) / (
                            max_value - min_value
                        )

                    else:

                        input_array = np.zeros_like(
                            input_array
                        )


            input_array = np.clip(
                input_array,
                0.0,
                1.0
            )


            # ------------------------------------------------
            # Display NPY information
            # ------------------------------------------------

            st.write(
                f"**NPY shape:** {original_shape}"
            )

            st.write(
                f"**NPY value range:** "
                f"{original_min:.6f} – "
                f"{original_max:.6f}"
            )


            # ------------------------------------------------
            # Convert NPY to PIL image
            # ------------------------------------------------

            image = Image.fromarray(
                (
                    input_array * 255.0
                ).clip(
                    0,
                    255
                ).astype(
                    np.uint8
                )
            )


        # ====================================================
        # READ PNG / JPG / JPEG
        # ====================================================

        else:

            image = Image.open(
                uploaded_file
            ).convert("L")

            input_array = (
                np.asarray(
                    image,
                    dtype=np.float32
                ) / 255.0
            )


        # ====================================================
        # RESIZE INPUT TO 128 × 128
        # ====================================================

        image_128 = image.resize(
            (128, 128),
            Image.Resampling.BICUBIC
        )


        # ====================================================
        # FINAL 128 × 128 INPUT ARRAY
        # ====================================================

        input_array = (
            np.asarray(
                image_128,
                dtype=np.float32
            ) / 255.0
        )


        # ====================================================
        # CONVERT TO PYTORCH TENSOR
        # ====================================================

        input_tensor = torch.from_numpy(
            input_array
        ).unsqueeze(0).unsqueeze(0)

        input_tensor = input_tensor.to(
            device
        )


        # ====================================================
        # LRDRN RESTORATION
        # ====================================================

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


        # ====================================================
        # PROCESSING TIME
        # ====================================================

        processing_time = (
            end_time - start_time
        )


        # ====================================================
        # CONVERT LRDRN OUTPUT TO NUMPY
        # ====================================================

        output_array = (
            restored[0, 0]
            .cpu()
            .numpy()
        )


        # ====================================================
        # BICUBIC BASELINE
        # ====================================================

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

        st.header(
            "🖼️ Image Comparison"
        )

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "Input"
            )

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

            st.subheader(
                "Bicubic"
            )

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

            st.subheader(
                "LRDRN"
            )

            st.image(
                output_array,
                caption="256 × 256 Restored",
                clamp=True,
                use_container_width=True
            )


        # ====================================================
        # OUTPUT INFORMATION
        # ====================================================

        st.header(
            "📋 Output Information"
        )

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

        st.subheader(
            "⚙️ Processing Information"
        )

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
        # DOWNLOAD SECTION
        # ====================================================

        st.header(
            "⬇️ Download Result"
        )


        # ----------------------------------------------------
        # PNG DOWNLOAD
        # ----------------------------------------------------

        output_uint8 = (
            output_array * 255.0
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )


        output_image = Image.fromarray(
            output_uint8
        )


        png_buffer = io.BytesIO()


        output_image.save(
            png_buffer,
            format="PNG"
        )


        st.download_button(
            label="⬇️ Download Restored PNG",
            data=png_buffer.getvalue(),
            file_name="LRDRN_restored.png",
            mime="image/png"
        )


        # ----------------------------------------------------
        # NPY DOWNLOAD
        # ----------------------------------------------------

        npy_buffer = io.BytesIO()


        np.save(
            npy_buffer,
            output_array
        )


        st.download_button(
            label="⬇️ Download Restored NPY",
            data=npy_buffer.getvalue(),
            file_name="LRDRN_restored.npy",
            mime="application/octet-stream"
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
            "❌ Error while processing the uploaded file."
        )

        st.exception(e)


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.header(
    "📚 About the Project"
)


# ------------------------------------------------------------
# ABOUT LRDRN
# ------------------------------------------------------------

with st.expander(
    "🧠 About LRDRN"
):

    st.write(
        """
        LRDRN (Lightweight Residual Dense Restoration Network)
        is used in this project for grayscale image
        super-resolution.

        The network receives a 128 × 128 grayscale image
        and produces a 256 × 256 restored image.
        """
    )


# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------

with st.expander(
    "📊 Model Performance"
):

    st.write(
        """
        **Improved LRDRN benchmark results**

        PSNR: **26.7967 dB**

        SSIM: **0.6811**

        Best Epoch: **38**
        """
    )


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

with st.expander(
    "📁 Dataset"
):

    st.write(
        """
        Training images: **2,560**

        Validation images: **640**

        Test images: **400**

        Image type: **Grayscale**
        """
    )


# ------------------------------------------------------------
# METHODOLOGY
# ------------------------------------------------------------

with st.expander(
    "⚙️ Methodology"
):

    st.write(
        """
        1. Upload a grayscale PNG, JPG, JPEG, or NPY file.

        2. The input is converted to grayscale.

        3. The input is resized to 128 × 128.

        4. The trained LRDRN model performs restoration.

        5. The output is generated at 256 × 256.

        6. Bicubic interpolation is generated as a baseline.

        7. The LRDRN restored image can be downloaded
           as PNG or NPY.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LRDRN Image Restoration Dashboard | KLA Dataset"
)
