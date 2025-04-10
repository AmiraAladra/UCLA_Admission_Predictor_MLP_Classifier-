# 🎓 UCLA Admission Predictor (MLP Classifier)
Project Overview
This project uses a Multi-layer Perceptron (MLP) neural network classifier to predict a student's admission chance to UCLA based on academic and personal profile attributes like GRE, TOEFL, GPA, SOP/LOR strength, and research experience.

The app offers an intuitive interface where users can enter their profile, receive a prediction, view model diagnostics, and explore visual insights about the admission dataset.

👉 Live App: 
https://ucla-admission-predictor-mlp-classifier.streamlit.app/

## 🚀 Features
Interactive Prediction: Users enter GRE, TOEFL, CGPA, SOP, LOR, and research experience to get instant predictions.

Loss Curve Visualization: See the training loss curve of the MLP model for transparency.

Admission Data Insights: View pre-generated visualizations including GRE vs TOEFL, CGPA distribution, and feature pairplots.

Error Handling: Gracefully handles missing models or figures with user-friendly messages.

Modular Codebase: Structured into src/ modules for easy maintenance and reusability.

## 📦 Dataset
The dataset used for training is Admission.csv, containing the following features:

Column Name	Description
Serial_No	Unique ID for each student (dropped during cleaning)
GRE_Score	Graduate Record Exam score (out of 340)
TOEFL_Score	Test of English as a Foreign Language (out of 120)
University_Rating	Rating of the university (1–5)
SOP	Statement of Purpose strength (1.0–5.0)
LOR	Letter of Recommendation strength (1.0–5.0)
CGPA	Undergraduate GPA (out of 10)
Research	1 if the student has research experience, else 0
Admit_Chance	Chance of admission (1 if ≥ 0.8, otherwise 0)
🛠 Technologies Used
Python 3.x

## 📚 Libraries:
pandas, numpy: Data handling and preprocessing

scikit-learn: Model training and evaluation

matplotlib, seaborn: Data visualization

streamlit: Web application

pickle: Model saving/loading

logging: Runtime tracking and error handling

## 🔍 Code Explanation
load_and_clean_data(path): Loads the raw CSV, drops unnecessary columns, and binarizes the target.

Create_dummy_variables(df): Encodes categorical variables and splits data into features and target.

train_MLPmodel(x, y): Trains a neural network (MLP), scales inputs, and saves the model.

evaluate_model(model, X_test, y_test): Computes accuracy and confusion matrix.

plot_loss_curve(model): Visualizes the training loss per epoch.

plot_scores(df): Saves GRE vs TOEFL scatter plot by admission status.

plot_cgpa_histogram_by_admission(df): Saves a CGPA histogram colored by admission.

plot_pairplot(df): Saves a pairplot of GRE, TOEFL, and CGPA.

## 🌐 Streamlit App Features
Sidebar Form Inputs:

GRE Score

TOEFL Score

University Rating

SOP & LOR Strength

CGPA

Research Experience

After Clicking Predict:

Displays prediction (Admit/Not Admit)

Shows admission probability

Displays model’s training loss curve

Shows saved graphs for dataset insights

## 📁 Project Structure

    ├── app.py                        # Streamlit web app
    ├── models/
    │   └── MLP.pkl                   # Trained MLP model
    ├── data/
    │   └── raw/
    │       └── Admission.csv         # Input dataset
    ├── logs/
    │   └── pipeline.log              # Runtime logs
    ├── reports/
    │   └── figures/
    │       ├── gre_vs_toefl.png
    │       ├── cgpa_hist.png
    │       └── pairplot.png
    ├── src/
    │   ├── data/
    │   │   └── data_processing.py
    │   ├── features/
    │   │   └── features_eng.py
    │   ├── models/
    │   │   ├── train_model.py
    │   │   └── predict_model.py
    │   └── visualization/
    │       └── visualize.py
    ├── requirements.txt
    └── README.md
##🖥 Installation (For Local Deployment)
1. Clone the Repository

  git clone https://github.com/yourusername/ucla-admission-predictor.git
  cd ucla-admission-predictor
2. Install Dependencies

  pip install -r requirements.txt
3. Train Model & Generate Plots
  python src/main.py
This will:
Train the MLP model and save it to models/MLP.pkl
Generate plots in reports/figures/

4. Run the App

  streamlit run app.py
  
##📈 Output Files
  models/MLP.pkl: Trained MLP model
  
  reports/figures/gre_vs_toefl.png: GRE vs TOEFL Scatter
  
  reports/figures/cgpa_hist.png: CGPA Histogram
  
  reports/figures/pairplot.png: Pairplot of main features

##🙌 Thank You!
Thanks for checking out the UCLA Admission Predictor!
Feel free to ⭐ the repo, contribute, raise issues, or share feedback.
