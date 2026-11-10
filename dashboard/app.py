import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2.keras"
COMPARISON_PATH = PROJECT_ROOT / "dataset" / "model_comparison.csv"

CLASS_NAMES = [
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy"
]

st.set_page_config(
    page_title="Potato AI | Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. CUSTOM DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F4F8F4;
}

/* Main page text - Ensures all general text is dark */
.main .stMarkdown p,
.main .stMarkdown li,
.main .stMarkdown strong,
.main p,
.main span,
.main [data-testid="stCaptionContainer"],
.main [data-testid="stWidgetLabel"] p,
.main [data-testid="stText"] {
    color: #202820 !important;
}

/* Main page headings */
.main h1,
.main h2,
.main h3,
.main h4 {
    color: #164B35 !important;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #123C2D;
}

/* Sidebar text - Ensures all text is white against dark background */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
}

/* Sidebar navigation */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #FFFFFF !important;
}

/* Main page title */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #164B35 !important;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    color: #46564B !important;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Hero banner - Dark background, white text */
.hero {
    background: linear-gradient(120deg, #145C3B, #238B57);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h2 {
    color: #FFFFFF !important;
    font-size: 28px;
    margin-bottom: 8px;
}

.hero p {
    color: #FFFFFF !important;
    font-size: 16px;
}

/* Metric cards - White background, dark text */
.metric-card {
    background-color: #FFFFFF;
    padding: 22px;
    border-radius: 15px;
    border-left: 5px solid #238B57;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
}

.metric-card .metric-label {
    color: #35483A !important;
    font-size: 14px;
}

.metric-card .metric-value {
    color: #164B35 !important;
    font-size: 27px;
    font-weight: 800;
}

/* Section titles */
.section-title {
    color: #164B35 !important;
    font-size: 23px;
    font-weight: 750;
    margin-top: 20px;
}

/* Prediction result card - Light green background, dark text */
.result-card {
    background-color: #E5F5E9;
    padding: 22px;
    border-radius: 15px;
    border-left: 5px solid #238B57;
}

.result-card .metric-label {
    color: #35483A !important;
}

.result-card .metric-value {
    color: #164B35 !important;
}

/* Streamlit information and warning boxes - Light background, dark text */
[data-testid="stAlert"] p,
[data-testid="stAlert"] strong,
[data-testid="stAlert"] div {
    color: #202820 !important;
}

/* File uploader - White background, dark text */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #202820 !important;
}

/* Buttons - Dark green background, white text */
div.stButton > button {
    background-color: #238B57;
    color: #FFFFFF !important;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-weight: 700;
}

div.stButton > button:hover {
    background-color: #145C3B;
    color: #FFFFFF !important;
}

/* Streamlit metrics */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #202820 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background-color: #238B57;
}

/* Footer */
.footer {
    text-align: center;
    color: #35483A !important;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD MODEL AND RESULTS
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🌱 Potato AI")

    st.write("Deep Learning Disease Detection")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "About Project",
            "Dashboard",
            "Disease Detection",
            "Model Performance"
        ],
        index=0
    )

    st.divider()

    st.markdown("### Model Information")
    st.write("**Model:** MobileNetV2")
    st.write("**Input size:** 224 × 224")
    st.write("**Classes:** 3")

    st.divider()

    st.caption("Potato Leaf Disease Detection Project")


# --------------------------------------------------
# 5. LOAD PROJECT FILES
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error("The MobileNetV2 model file could not be found.")
    st.stop()

if not COMPARISON_PATH.exists():
    st.error("The model comparison CSV file could not be found.")
    st.stop()

try:
    model = load_model()
    comparison_df = load_comparison()

except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


# --------------------------------------------------
# 6. ABOUT PROJECT - FIRST PAGE
# --------------------------------------------------

if page == "About Project":

    st.markdown(
        '<div class="main-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Project Overview

    Potato Leaf Disease Detection applies deep learning and image
    classification to identify three categories of potato leaves.

    The project investigates how convolutional neural networks
    and transfer learning can support automated crop monitoring.

    ### Why This Project?

    Potato diseases can affect crop health and agricultural
    productivity. Identifying disease symptoms from leaf images
    provides an opportunity to explore how deep learning can
    support crop monitoring.

    ### Dataset

    The dataset was obtained from the PlantVillage repository.

    The three categories are:

    - Potato Early Blight
    - Potato Late Blight
    - Potato Healthy

    ### Models Evaluated

    **1. Baseline CNN**

    A custom Convolutional Neural Network trained to identify
    patterns in potato leaf images.

    **2. MobileNetV2**

    A transfer-learning model using pretrained ImageNet features
    and a custom classification layer.

    ### Best Model

    MobileNetV2 achieved a test accuracy of 96.90% on the
    held-out test dataset.

    ### Limitations

    The PlantVillage images were collected under controlled
    conditions. Images taken directly from farms may have
    different lighting, backgrounds and leaf conditions.

    The model has not been validated for real-world farm deployment.
    Its predictions should not replace expert agricultural advice.

    ### Dataset Source

    [PlantVillage Dataset on GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
    """)


# --------------------------------------------------
# 7. DASHBOARD HOME
# --------------------------------------------------

elif page == "Dashboard":

    st.markdown(
        '<div class="main-title">Potato Leaf Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">An intelligent agriculture dashboard '
        'powered by deep learning.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h2>Smarter Crop Monitoring Starts Here</h2>
        <p>
        Upload a potato leaf image, explore AI predictions and
        compare deep learning model performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Disease Categories</div>
            <div class="metric-value">03</div>
            <div class="metric-label">
                Early blight, late blight and healthy
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value">MobileNetV2</div>
            <div class="metric-label">Transfer learning</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        accuracy = comparison_df.loc[
            comparison_df["Model"] == "MobileNetV2",
            "Test Accuracy"
        ].iloc[0]

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">{accuracy:.2%}</div>
            <div class="metric-label">
                Performance on held-out test data
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Explore the Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Disease Detection**\n\n"
            "Upload a potato leaf image and view the model's prediction."
        )

    with col2:
        st.info(
            "**Model Performance**\n\n"
            "Explore how the baseline CNN compares with MobileNetV2."
        )


# --------------------------------------------------
# 8. DISEASE DETECTION
# --------------------------------------------------

elif page == "Disease Detection":

    st.markdown(
        '<div class="main-title">Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image of a potato leaf to classify its condition."
    )

    uploaded_file = st.file_uploader(
        "Choose a potato leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(
                    image,
                    caption="Uploaded Potato Leaf",
                    use_container_width=True
                )

            with col2:

                st.markdown("### Image Information")

                st.write(f"**Image width:** {image.width} pixels")
                st.write(f"**Image height:** {image.height} pixels")
                st.write("**Model input:** 224 × 224 pixels")

                if st.button(
                    "Analyse Leaf Image",
                    use_container_width=True
                ):

                    with st.spinner("Analysing image..."):

                        resized_image = image.resize((224, 224))

                        image_array = np.array(resized_image)

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        image_array = preprocess_input(
                            image_array.astype(np.float32)
                        )

                        predictions = model.predict(
                            image_array,
                            verbose=0
                        )[0]

                        predicted_index = int(np.argmax(predictions))

                        predicted_class = CLASS_NAMES[predicted_index]

                        confidence = float(
                            predictions[predicted_index]
                        ) * 100

                    st.session_state["prediction"] = predicted_class
                    st.session_state["confidence"] = confidence
                    st.session_state["probabilities"] = predictions
                    st.session_state["analysed_file"] = uploaded_file.name

            # Display prediction results

            if (
                "prediction" in st.session_state
                and st.session_state.get("analysed_file") == uploaded_file.name
            ):

                st.divider()

                st.markdown("### Prediction Results")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="metric-label">Predicted Condition</div>
                        <div class="metric-value">
                            {st.session_state["prediction"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_col2:

                    st.metric(
                        "Model Confidence",
                        f'{st.session_state["confidence"]:.2f}%'
                    )

                st.markdown("### Class Probability Breakdown")

                probabilities = st.session_state["probabilities"]

                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities
                ):

                    percentage = float(probability) * 100

                    st.write(
                        f"**{class_name} — {percentage:.2f}%**"
                    )

                    st.progress(float(probability))

                st.warning(
                    "This is an experimental prediction from a model "
                    "trained on PlantVillage images. It is not a "
                    "professional agricultural diagnosis."
                )

        except Exception as error:
            st.error(f"Unable to process this image: {error}")

    else:

        st.info(
            "Upload a JPG, JPEG or PNG image to begin disease detection."
        )


# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare the test performance of the two trained models."
    )

    st.markdown("### Model Comparison")

    styled_comparison = comparison_df.style.format({
        "Test Accuracy": "{:.2%}",
        "Test Loss": "{:.4f}",
        "Macro F1 Score": "{:.4f}",
        "Weighted F1 Score": "{:.4f}"
    }).set_properties(**{
        "color": "#202820",
        "background-color": "#FFFFFF"
    }).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("color", "#FFFFFF"),
                ("background-color", "#164B35"),
                ("font-weight", "bold")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("color", "#202820"),
                ("background-color", "#FFFFFF")
            ]
        }
    ])

    st.dataframe(
        styled_comparison,
        use_container_width=True,
        hide_index=True
    )

    # Accuracy chart

    st.markdown("### Test Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df["Test Accuracy"] * 100,
        color=["#91C9A8", "#238B57"],
        width=0.5
    )

    ax.set_ylabel("Test Accuracy (%)", color="#202820")
    ax.set_title(
        "Baseline CNN vs MobileNetV2",
        color="#164B35"
    )
    ax.set_ylim(0, 110)

    ax.tick_params(axis="x", colors="#202820")
    ax.tick_params(axis="y", colors="#202820")

    ax.bar_label(
        bars,
        fmt="%.2f%%",
        padding=4,
        color="#202820"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # F1 score comparison

    st.markdown("### F1 Score Comparison")

    f1_columns = [
        "Macro F1 Score",
        "Weighted F1 Score"
    ]

    f1_data = comparison_df.set_index("Model")[f1_columns]

    st.bar_chart(f1_data)

    st.success(
        "MobileNetV2 achieved the highest test accuracy "
        "among the two evaluated models."
    )

    st.caption(
        "Results are based on the held-out test dataset."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="footer">
        Potato AI | Potato Leaf Disease Detection<br>
        Deep Learning Academic Project
    </div>
    """,
    unsafe_allow_html=True
)