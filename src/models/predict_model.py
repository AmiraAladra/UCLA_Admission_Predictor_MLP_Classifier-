# import evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import logging

# Function to predict and evaluate
def evaluate_model(MLP, xtest_scaled, ytest):
    try:
        logging.info("Making predictions with MLP model...")
        ypred = MLP.predict(xtest_scaled)

        logging.info("Calculating confusion matrix and accuracy...")
        confusion_mat = confusion_matrix(ytest, ypred)
        accuracy = accuracy_score(ytest, ypred)

        logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy, confusion_mat

    except Exception as e:
        logging.error("Error occurred during model evaluation", exc_info=True)
        raise
