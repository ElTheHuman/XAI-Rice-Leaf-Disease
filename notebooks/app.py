import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model


# =============================================================================
# APP CONFIG
# =============================================================================
st.set_page_config(page_title="Rice Leaf Classifier + LRP", page_icon="🌾", layout="centered")


# =============================================================================
# PATHS + LABELS
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
WEIGHT_PATH = (APP_DIR / ".." / "src" / "models" / "res_net_model_weight.h5").resolve()

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Sheath Blight",
]


# Optional: show logo/loading gif if you have them
LOGO_PATH = APP_DIR / "assets" / "logo.png"
LOADING_GIF = APP_DIR / "assets" / "loading.gif"


def toast(msg: str, icon: str = "✅"):
    """Use toast if available; otherwise do nothing (keeps compatibility)."""
    if hasattr(st, "toast"):
        st.toast(msg, icon=icon)


# =============================================================================
# MODEL
# =============================================================================
def build_model(num_classes: int = 7) -> Model:
    """ResNet50 backbone (frozen) + small classifier head."""
    inp = Input(shape=(256, 256, 3))
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


@st.cache_resource(show_spinner=False)
def load_model_cached(weight_path: str) -> Model:
    """Load once per session (prevents reload on every Streamlit rerun)."""
    m = build_model(num_classes=len(CLASS_NAMES))
    m.load_weights(weight_path)
    return m


# =============================================================================
# IMAGE PREP
# =============================================================================
def preprocess_uploaded_image(uploaded_file):
    """
    Returns:
      img_batch : (1, 256, 256, 3) preprocessed for ResNet
      raw_img   : (256, 256, 3) raw image array for display
    """
    img = image.load_img(uploaded_file, target_size=(256, 256))
    raw_img = image.img_to_array(img)  # float32 in [0..255]

    img_pre = preprocess_input(raw_img.copy())
    img_batch = np.expand_dims(img_pre, axis=0)

    return img_batch, raw_img


# =============================================================================
# PREDICT
# =============================================================================
def predict(model: Model, img_batch: np.ndarray):
    preds = model.predict(img_batch, verbose=0)[0]  # (7,)
    idx = int(np.argmax(preds))
    return preds, idx


# =============================================================================
# LRP (single image)
# =============================================================================
def generate_lrp_single(img_batch: np.ndarray):
    """
    Generates a 2D relevance map scaled to [-1, 1].
    NOTE: innvestigate often prefers graph mode; we keep your approach.
    """
    import innvestigate
    import innvestigate.analyzer.relevance_based.relevance_rule as rrule
    import innvestigate.analyzer.relevance_based.relevance_analyzer as ranalyzer
    import tensorflow.keras.layers as klayers

    # Keep your original behavior (but safer)
    if tf.executing_eagerly():
        try:
            tf.compat.v1.disable_eager_execution()
        except Exception:
            # If TF complains, you can move disable_eager_execution() to the top of the file.
            pass

    model_lrp = build_model(num_classes=len(CLASS_NAMES))
    model_lrp.load_weights(str(WEIGHT_PATH))

    # Remove softmax for relevance computation
    try:
        model_wo_sm = innvestigate.utils.model_wo_softmax(model_lrp)
    except Exception:
        model_wo_sm = model_lrp

    class EpsilonProxyRule(rrule.EpsilonRule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, epsilon=0.1, bias=True, **kwargs)

    CONV_LAYERS = (klayers.Conv2D, klayers.Conv3D, klayers.SeparableConv2D, klayers.DepthwiseConv2D)
    PASS_LAYERS = (
        klayers.Activation, klayers.Flatten, klayers.Dropout,
        klayers.Add, klayers.MaxPooling2D, klayers.GlobalAveragePooling2D
    )

    rules = [
        (lambda l: isinstance(l, klayers.Dense), EpsilonProxyRule),
        (lambda l: isinstance(l, CONV_LAYERS), rrule.Alpha2Beta1Rule),
        (lambda l: isinstance(l, PASS_LAYERS), "Pass"),
    ]

    analyzer = ranalyzer.LRP(
        model_wo_sm,
        rule=rules,
        input_layer_rule=rrule.FlatRule,
        bn_layer_rule=rrule.AlphaBetaX2m100Rule,
    )

    # single image -> (256,256,3)
    rel = analyzer.analyze(img_batch)[0]

    # Your normalization pipeline (cleaned + safe)
    rel_map = rel.sum(axis=-1)
    vmax = np.percentile(np.abs(rel_map), 99)
    vmax = max(vmax, 1e-8)  # avoid divide-by-zero
    rel_map = np.clip(rel_map, -vmax, vmax) / vmax  # -> [-1, 1]

    return rel_map


# =============================================================================
# UI
# =============================================================================
st.title("🌾 Rice Leaf Classification + LRP (Single Image)")

# Sidebar “branding” + run indicator
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown("### Status")
    sidebar_status = st.empty()

# Validate weights early
if not WEIGHT_PATH.exists():
    st.error(f"Model weights not found:\n{WEIGHT_PATH}")
    st.stop()

# Load model with visible feedback
sidebar_status.info("Loading model…")
loading_slot = st.empty()

if LOADING_GIF.exists():
    loading_slot.image(str(LOADING_GIF), caption="Loading…", use_container_width=True)

with st.spinner("Loading the model (cached)…"):
    model = load_model_cached(str(WEIGHT_PATH))

loading_slot.empty()
sidebar_status.success("Model ready ✅")
toast("Model loaded", "✅")


st.markdown("Upload an image of a rice leaf, then generate a prediction and an LRP relevance map.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_batch, raw_img = preprocess_uploaded_image(uploaded)

    colA, colB = st.columns([1, 1])

    with colA:
        st.image(raw_img.astype(np.uint8), caption="Uploaded Image", use_container_width=True)

    # Prediction block with feedback
    with colB:
        with st.spinner("Running prediction…"):
            preds, idx = predict(model, img_batch)

        st.subheader("Prediction")
        st.write(f"**{CLASS_NAMES[idx]}**")

        topk = np.argsort(preds)[::-1][:3]
        st.caption("Top 3 probabilities:")
        for i in topk:
            st.write(f"- {CLASS_NAMES[i]}: **{preds[i]*100:.2f}%**")

        # Small bar chart (no forced colors)
        figp, axp = plt.subplots(figsize=(4, 2.8))
        axp.bar([CLASS_NAMES[i] for i in topk], [preds[i] for i in topk])
        axp.set_ylim(0, 1)
        axp.set_ylabel("Probability")
        axp.tick_params(axis="x", rotation=15)
        st.pyplot(figp)

    st.divider()

    st.subheader("LRP Explainability")
    overlay = st.checkbox("Overlay relevance map on the original image", value=True)

    if st.button("Generate LRP Map"):
        toast("LRP running…", "🧠")
        sidebar_status.warning("Computing LRP…")

        progress = st.progress(0, text="Preparing LRP…")
        try:
            progress.progress(20, text="Building analyzer…")
            with st.spinner("Computing relevance map…"):
                progress.progress(60, text="Analyzing image…")
                rel_map = generate_lrp_single(img_batch)
                progress.progress(90, text="Rendering…")

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            if overlay:
                ax.imshow(raw_img.astype(np.uint8) / 255.0)
                ax.imshow(rel_map, cmap="seismic", vmin=-1, vmax=1, alpha=0.5)
                ax.set_title("LRP Relevance (Overlay)")
            else:
                ax.imshow(rel_map, cmap="seismic", vmin=-1, vmax=1)
                ax.set_title("LRP Relevance Map")

            ax.axis("off")
            st.pyplot(fig)

            progress.progress(100, text="Done.")
            sidebar_status.success("LRP complete ✅")
            toast("LRP complete", "✅")

        except Exception as e:
            sidebar_status.error("LRP failed ❌")
            st.error(f"LRP error: {e}")
        finally:
            progress.empty()

with st.expander("How this app works (short explanation)"):
    st.write(
        "1) The uploaded image is resized to 256×256.\n"
        "2) ResNet preprocessing is applied (same preprocessing used during training).\n"
        "3) The model outputs 7 class probabilities (softmax).\n"
        "4) LRP backpropagates relevance from the output back to pixels, producing a heatmap.\n"
        "   Red/blue show positive/negative contributions after normalization to [-1, 1]."
    )
