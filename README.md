# 🩺 Comparative Performance Analysis of Deep Learning Models in Cardiovascular Diseases

---

### 📘 Overview
An *AI-powered Deep Learning project* designed to classify *cardiovascular diseases (CVDs)* from *ECG (Electrocardiogram) images*.  
The project compares the performance of *Convolutional Neural Networks (CNN)* and *Recurrent Neural Networks (RNN)* to determine which architecture provides superior accuracy and efficiency for ECG-based diagnosis.

This system aims to support *early detection* and *automated screening* of heart abnormalities, reducing the dependency on manual ECG interpretation and minimizing human error.

---

### 🎯 Project Objectives
To provide *automated and data-driven diagnosis* of cardiovascular diseases through:
- ⚙ Implementation of *CNN and RNN models* for ECG image classification  
- 📊 *Comparative analysis* based on accuracy, precision, recall, and F1-score  
- 🧠 *Feature extraction and preprocessing* using CLAHE, edge detection, and image augmentation  
- 🌐 Development of a *Flask-based web interface* for real-time predictions  
- 🔍 *Model interpretability* through Grad-CAM heatmaps for decision transparency  

---

### 🧩 Methodology
1. *Data Collection:* Publicly available ECG datasets (e.g., MIT-BIH Arrhythmia Database)  
2. *Preprocessing:* Noise removal, grayscale conversion, normalization, and augmentation  
3. *Feature Extraction:* Morphological and frequency-based ECG characteristics  
4. *Model Development:*  
   - *CNN (EfficientNetB0):* For spatial ECG feature extraction  
   - *RNN (EfficientNet + LSTM):* For sequential pattern analysis  
5. *Evaluation Metrics:* Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
6. *Deployment:* Flask web app for ECG image upload and disease prediction  

---

### 💻 Tech Stack
| Category | Tools / Frameworks |
|-----------|--------------------|
| *Programming Language* | Python 3.8+ |
| *Deep Learning Frameworks* | TensorFlow / Keras, PyTorch |
| *Data Processing* | NumPy, Pandas, Scikit-learn |
| *Visualization* | Matplotlib, Seaborn |
| *ECG Analysis* | BioSPPy, WFDB, NeuroKit2 |
| *Web Framework* | Flask |
| *Frontend* | HTML, CSS, Bootstrap |
| *Version Control* | Git, GitHub |

---

### 📈 Results & Findings
| Model | Accuracy | Key Insight |
|--------|-----------|-------------|
| *CNN (EfficientNetB0)* | >90% | Excellent at identifying spatial ECG features |
| *RNN (EfficientNet + LSTM)* | ~70% | Less effective for image data but captures temporal dependencies |
| *Conclusion:* CNN outperforms RNN for ECG image-based cardiovascular disease detection |

---

### 🧠 Model Visualization
- *Grad-CAM* used to highlight ECG regions influencing predictions  
- Improves interpretability and helps validate diagnostic reliability  

---

### 🌐 Web Application
A *Flask-based web app* allows users to:
- Upload ECG images  
- View predicted results with condition name, cause, and treatment suggestions  
- Visualize Grad-CAM overlays for interpretability  

*Routes:*
- / → Home page  
- /disease-predict → Upload and prediction page  

---

### 🚀 Future Scope
- Develop *hybrid CNN–RNN architectures* for spatio-temporal learning  
- Integrate with *IoT-enabled ECG monitoring devices* for real-time detection  
- Deploy as a *mobile health application* for patient use  
- Explore *multi-modal medical datasets* (signals + images) for improved accuracy  
- Incorporate *transfer learning and federated learning* for scalable cloud deployment  

---
🏠 1. Home Page – “AI-Powered Arrhythmia Detection System”

<img width="1914" height="932" alt="Screenshot 2025-04-25 123641" src="https://github.com/user-attachments/assets/014b7325-5b33-4d78-905d-d2b2a0824266" />
Content (Description):
- Welcome to our Arrhythmia Detection System, an AI-based healthcare application that utilizes deep learning algorithms to classify ECG signals and detect heart rhythm abnormalities.
- Our goal is to assist healthcare professionals in early detection of cardiac arrhythmias, ensuring faster diagnosis and improved patient outcomes.

Key Highlights:
- Built using Convolutional Neural Networks (CNNs) trained on ECG datasets.
- Provides real-time classification of uploaded ECG images.
- Designed with a simple and user-friendly interface.
- A step toward AI-driven cardiac diagnostics.

📤 2. Upload Page – “ECG Image Upload & Prediction” 

<img width="1919" height="949" alt="Uploading File" src="https://github.com/user-attachments/assets/43113092-19dc-4a47-8333-f28b0435d08c" />
Content (Description):
- Upload a clear ECG image to analyze your heart rhythm pattern using our AI model.
- The system processes the image through a trained deep learning model to detect whether the signal represents a normal heartbeat or a type of arrhythmia.

Instructions to Use:
- Click the “Choose File” button and upload your ECG image (PNG/JPG format).
- Ensure the ECG image is clear and noise-free.
- Click “Predict” to begin the analysis.
- The result will display within seconds with disease type, possible cause, and treatment advice.

Note:This tool is for research and educational purposes only. Always consult a cardiologist for clinical interpretation.

📊 3. Result Page – “ECG Classification Result”

<img width="1898" height="933" alt="Result" src="https://github.com/user-attachments/assets/d7726e2d-5c0a-40d4-8319-7f9d70bf4cf2" />
Content (Description):
Based on the uploaded ECG image, the deep learning model predicts the disease category. 

Each result page provides detailed information including:
- Disease Name: (e.g., Normal, Arrhythmia, Myocardial Infarction, etc.)
- Cause: Explains the underlying rhythm or abnormality detected.
- Treatment/Recommendation: Suggests preventive or corrective actions.

Example Output:
Disease: N (Normal Heartbeat)
Cause: Standard heartbeat with normal rhythm and no abnormalities.
Treatment: No treatment required. Maintain a healthy lifestyle and regular check-ups.

Additional Features:
- Displays the uploaded ECG waveform alongside the classification result.
- Offers contextual medical explanation for each type of ECG pattern.
- Encourages users to seek professional advice for abnormal detections.
