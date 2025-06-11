
```markdown
# 🧠 Brain Tumor Detection Dashboard

This project is a web-based dashboard built using **Streamlit** that detects brain tumors in MRI images using a Convolutional Neural Network (CNN). It includes model training, evaluation, and a user interface for uploading MRI images and predicting whether a tumor is present.

---

## 🚀 Features

- Real-time dashboard visualization with Streamlit  
- CNN-based binary image classification (Tumor / No Tumor)  
- Interactive charts: Accuracy, Loss, Confusion Matrix, ROC Curve  
- Image upload functionality for instant prediction  
- Clean, professional UI with metrics and plots  

---

## 📂 Project Structure


## 🧑‍💻 Installation

### 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### 💻 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/BrainTumorScan.git
cd BrainTumorScan

# 2. (Optional but recommended) Create a virtual environment
py -3.8 -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app/app.py
````

---

## 🛠️ Fix for Protobuf Compatibility Error

If you encounter issues with `protobuf` versions while running your model, follow this:

### ✅ Step-by-Step Fix

With your virtual environment activated:

```bash
pip uninstall protobuf
pip install protobuf==3.20.3
```

Then verify the installed version:

```bash
pip show protobuf
```

You should see:

```
Version: 3.20.3
```

---

## 📈 Model Details

* **Architecture**: CNN with Conv2D, MaxPooling, Flatten, Dense, Dropout
* **Loss**: Binary Crossentropy
* **Optimizer**: Adam
* **Evaluation Metrics**: Accuracy, Confusion Matrix, ROC-AUC

---

## 📷 Prediction

After training, you can upload your own MRI image using the web interface. The model will classify it as either:

* 🔴 **Tumor Detected**
* 🟢 **No Tumor Detected**

> Make sure `models/brain_tumor_cnn.h5` exists after training. If not, re-run the training section in `app.py`.

---

## 📊 Sample Visualizations

* Accuracy/Loss over epochs
* Confusion Matrix
* ROC Curve
* Uploaded MRI image with prediction result

---

## 🧠 Dataset

The project assumes the dataset is stored locally in the following structure:

```
data/
├── yes/    # Tumor images
└── no/     # Normal images
```

You can get brain MRI datasets from sources like [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

---

## 🙌 Acknowledgments

* [TensorFlow](https://www.tensorflow.org/)
* [Streamlit](https://streamlit.io/)
* [Kaggle Datasets](https://www.kaggle.com/)

---

## 📃 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🔗 Connect

Built with ❤️ by Bharath C O
🚀 Let's innovate with AI and health-tech
📬 [LinkedIn](https://www.linkedin.com/in/bharath-c-o-226a1229a/)

````

---
