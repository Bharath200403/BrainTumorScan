import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# --- Streamlit Config ---
st.set_page_config(page_title="🧠 Brain Tumor Detector", layout="wide")
st.sidebar.title("🧠 Brain Tumor Detection")
st.sidebar.info("Upload MRI images and evaluate a CNN model")
st.sidebar.markdown("---")
st.sidebar.markdown("📌 Developed by Bharath C O")

st.title("🧠 Brain Tumor Detection Dashboard")
st.markdown("This dashboard trains a CNN model to classify brain MRI images into Tumor and No Tumor classes.")

# --- 1. Data Preparation ---
def load_images(folder, label, size=(128, 128)):
    images, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert('RGB').resize(size)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
        except:
            continue
    return images, labels

with st.spinner("Loading and processing images..."):
    tumor_images, tumor_labels = load_images('data/yes', 1)
    normal_images, normal_labels = load_images('data/no', 0)
    X = np.array(tumor_images + normal_images)
    y = np.array(tumor_labels + normal_labels)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- 2. Build and Train Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, monitor='val_loss'),
    ModelCheckpoint("models/brain_tumor_cnn.h5", save_best_only=True)
]

with st.spinner("Training the CNN model..."):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

# --- 3. Evaluation Metrics ---
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Accuracy Over Epochs")
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    st.pyplot(plt)

with col2:
    st.markdown("#### 📉 Loss Over Epochs")
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    st.pyplot(plt)

st.subheader("📌 Confusion Matrix")
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor'], output_dict=True)
st.dataframe(report)

st.subheader("📉 ROC Curve")
y_probs = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# --- 4. Predict New MRI Image ---
st.markdown("---")
st.header("🔍 Upload MRI Image for Prediction")
with st.expander("📤 Upload an Image", expanded=True):
    uploaded_file = st.file_uploader("Choose an MRI image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🧠 Uploaded MRI", use_column_width=True)

        image = image.resize((128, 128))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        model_path = "models/brain_tumor_cnn.h5"
        if os.path.exists(model_path):
            best_model = load_model(model_path)
            prediction = best_model.predict(image_array)[0][0]
            st.subheader("🧪 Prediction Result")
            if prediction > 0.5:
                st.error("🔴 Tumor Detected – Please consult a doctor.")
            else:
                st.success("🟢 No Tumor Detected – Image appears normal.")
        else:
            st.warning("⚠️ Model not found. Please ensure brain_tumor_cnn.h5 is in /models")
