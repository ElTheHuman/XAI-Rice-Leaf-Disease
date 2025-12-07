import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
import cv2

# =========================================================
# BUILD MODEL (SAMA PERSIS DENGAN TRAINING)
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
# LOAD MODEL UNTUK PREDIKSI (EAGER)
# =========================================================
WEIGHT_PATH = "../src/models/res_net_model_weight.h5"   # <--- EDIT SESUAI FILE KAMU

model = build_model()
model.load_weights(WEIGHT_PATH)

# Label class kamu (EDIT ya sayang)
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
# IMAGE PREPROCESSING
# =========================================================
def preprocess_uploaded_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(img)


# =========================================================
# PREDIKSI NORMAL
# =========================================================
def predict_image(img_array):
    preds = model.predict(img_array, verbose=1)
    idx = np.argmax(preds)
    return preds, idx


# =========================================================
# GENERATE LRP DARI MODEL WEIGHT (GRAPH MODE)
# =========================================================
def generate_lrp(img_array):

    import innvestigate
    import innvestigate.analyzer.relevance_based.relevance_rule as rrule
    import innvestigate.analyzer.relevance_based.relevance_analyzer as ranalyzer
    import tensorflow.keras.layers as klayers

    # Disable eager khusus untuk LRP
    tf.compat.v1.disable_eager_execution()

    # Build model lagi di graph mode
    model_lrp = build_model()
    model_lrp.load_weights(WEIGHT_PATH)

    # remove softmax (wajib)
    try:
        model_wo_sm = innvestigate.utils.model_wo_softmax(model_lrp)
    except:
        model_wo_sm = model_lrp

    # ============== CUSTOM RULES ==============
    class EpsilonProxyRule(rrule.EpsilonRule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, epsilon=0.1, bias=True, **kwargs)

    CONV_LAYERS = (
        klayers.Conv1D, klayers.Conv2D, klayers.Conv3D,
        klayers.SeparableConv2D, klayers.DepthwiseConv2D
    )

    PASS_LAYERS = (
        klayers.Activation, klayers.Flatten, klayers.Dropout,
        klayers.Add, klayers.MaxPooling2D, klayers.GlobalAveragePooling2D
    )

    rules = [
        (lambda layer: isinstance(layer, klayers.Dense), EpsilonProxyRule),
        (lambda layer: isinstance(layer, CONV_LAYERS), rrule.Alpha2Beta1Rule),
        (lambda layer: isinstance(layer, PASS_LAYERS), "Pass"),
    ]

    analyzer = ranalyzer.LRP(
        model_wo_sm,
        rule=rules,
        input_layer_rule=rrule.FlatRule,
        bn_layer_rule=rrule.AlphaBetaX2m100Rule
    )

    # LRP relevances
    relevance = analyzer.analyze(img_array)[0]

    # normalize
    heat = relevance.sum(axis=-1)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)

    heat_img = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return heat_img


# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🌾 Rice Leaf Classification + LRP (Weights Only)")

uploaded_file = st.file_uploader("Upload Image Daun", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_array, raw_img = preprocess_uploaded_image(uploaded_file)
    st.image(raw_img, caption="Uploaded Image", width=300)

    preds, idx = predict_image(img_array)
    st.subheader("Prediction")
    st.write(f"**{CLASS_NAMES[idx]}**")
    st.write("Probabilities:", preds[0])

    if st.button("Generate LRP"):
        st.write("Computing LRP… sabar sebentar ya sayang ❤️")
        heatmap = generate_lrp(img_array)
        st.image(heatmap, caption="LRP Heatmap", width=300)
        st.success("Done sayanggg 💗")
