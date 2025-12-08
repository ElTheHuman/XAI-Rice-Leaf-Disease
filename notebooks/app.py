import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# =========================================================
# BUILD MODEL
# =========================================================
def build_model():
    inp = Input(shape=(256, 256, 3))
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(7, activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


# =========================================================
# LOAD WEIGHTS
# =========================================================
WEIGHT_PATH = "../src/models/res_net_model_weight.h5"

model = build_model()
model.load_weights(WEIGHT_PATH)

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Sheath Blight"
]


# =========================================================
# PREPROCESS IMAGE
# =========================================================
def preprocess_uploaded_image(uploaded_file):
    # load original image dulu
    img = image.load_img(uploaded_file, target_size=(256, 256))
    raw_img = image.img_to_array(img)  # ini buat ditampilin

    # preprocess untuk model
    img_arr_pre = preprocess_input(raw_img.copy())  # jangan timpa raw_img
    img_arr_pre = np.expand_dims(img_arr_pre, axis=0)

    return img_arr_pre, raw_img  # preprocessed, original


# =========================================================
# PREDICT
# =========================================================
def predict(img_array):
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return preds, idx


# =========================================================
# LRP FOR ONE IMAGE ONLY
# =========================================================
def generate_lrp_single(img_array):
    import innvestigate
    import innvestigate.analyzer.relevance_based.relevance_rule as rrule
    import innvestigate.analyzer.relevance_based.relevance_analyzer as ranalyzer
    import tensorflow.keras.layers as klayers

    # disable eager for LRP only
    tf.compat.v1.disable_eager_execution()

    # build model in graph mode
    model_lrp = build_model()
    model_lrp.load_weights(WEIGHT_PATH)

    # remove softmax
    try:
        model_wo_sm = innvestigate.utils.model_wo_softmax(model_lrp)
    except:
        model_wo_sm = model_lrp

    # custom rules
    class EpsilonProxyRule(rrule.EpsilonRule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, epsilon=0.1, bias=True, **kwargs)

    CONV_LAYERS = (
        klayers.Conv2D, klayers.Conv3D, klayers.SeparableConv2D, klayers.DepthwiseConv2D
    )

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
        bn_layer_rule=rrule.AlphaBetaX2m100Rule
    )

    # hanya 1 image
    rel = analyzer.analyze(img_array)[0]   # shape (256,256,3)

    # EXACT CODE FROM YOU
    rel_map = rel.sum(axis=-1)                # reduce channels
    vmax = np.percentile(np.abs(rel_map), 99) # 99th percentile
    rel_map = np.clip(rel_map, -vmax, vmax)   # clip
    rel_map /= vmax                           # normalize to [-1,1]

    return rel_map  # 2D map


# =========================================================
# STREAMLIT APP
# =========================================================
st.title("🌾 Rice Leaf Classification + LRP (Single Image)")

uploaded = st.file_uploader("Upload Daun Padi", type=["jpg", "jpeg", "png"])

if uploaded:
    img_array, raw_img = preprocess_uploaded_image(uploaded)

    # tampilkan image asli (bukan preprocessed)
    st.image(raw_img.astype(np.uint8), caption="Uploaded Image", width=320)

    preds, idx = predict(img_array)

    st.subheader("Prediction")
    st.write(f"**{CLASS_NAMES[idx]}**")
    st.write(preds[0])

    if st.button("Generate LRP Map ❤️"):
        st.write("Lagi dihitung ya sayang… 💕")

        rel_map = generate_lrp_single(img_array)

        # overlay LRP di atas image asli (lebih visual)
        fig, ax = plt.subplots(figsize=(4, 4))
        # ax.imshow(raw_img.astype(np.uint8)/255.0)  # tampilkan asli
        # ax.imshow(rel_map, cmap="seismic", clim=(-1, 1), alpha=0.5)  # overlay
        ax.imshow(rel_map, cmap="seismic", clim=(-1, 1))  # overlay
        ax.set_title("LRP Relevance Map")
        ax.axis("off")

        st.pyplot(fig)

        st.success("Done sayangg 💗")
