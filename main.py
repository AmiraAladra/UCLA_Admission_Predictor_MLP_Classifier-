import logging
from src.data.data_processing import load_and_clean_data
from src.features.features_eng import Create_dummy_variables
from src.models.train_model import train_MLPmodel
from src.models.predict_model import evaluate_model
from src.visualization.visualize import plot_scores, plot_scores, plot_pairplot, plot_cgpa_histogram_by_admission


logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    try:
        logging.info("Starting the admission prediction pipeline...")

        # Load and preprocess the data
        data_path = "data/raw/Admission.csv"
        logging.info(f"Loading dataset from: {data_path}")
        df = load_and_clean_data(data_path)

        # Plot and save GRE vs TOEFL scores
        logging.info("Generating scatter plot of GRE vs TOEFL scores")
        plot_scores(df)
        plot_pairplot(df)
        plot_cgpa_histogram_by_admission(df)

        # Create dummy variables and separate features and target
        logging.info("Creating dummy variables and splitting features/target")
        x, y = Create_dummy_variables(df)

        # Train the model
        logging.info("Training the MLP model")
        MLP, xtest_scaled, ytest = train_MLPmodel(x, y)

        # Evaluate the model
        logging.info("Evaluating the model")
        accuracy, confusion_mat = evaluate_model(MLP, xtest_scaled, ytest)

        # Output results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{confusion_mat}")
        logging.info(f"Model Accuracy: {accuracy:.4f}")
        logging.info(f"Confusion Matrix:\n{confusion_mat}")

    except Exception as e:
        logging.error("An error occurred in the pipeline.", exc_info=True)
        print("❌ An error occurred. Please check the log file at logs/pipeline.log.")

    
    