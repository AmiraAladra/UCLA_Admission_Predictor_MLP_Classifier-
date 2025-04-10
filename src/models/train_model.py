from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import pickle
import logging

def train_MLPmodel(x, y):
    try:
        logging.info("Splitting dataset into train and test sets...")
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)

        logging.info("Fitting MinMaxScaler on training data...")
        scaler = MinMaxScaler()
        scaler.fit(xtrain)

        logging.info("Transforming training and test data using scaler...")
        xtrain_scaled = scaler.transform(xtrain)
        xtest_scaled = scaler.transform(xtest)

        logging.info("Training MLP classifier...")
        MLP = MLPClassifier(hidden_layer_sizes=(2, 2), batch_size=50, max_iter=200)
        MLP.fit(xtrain_scaled, ytrain)

        logging.info("Saving trained MLP model to 'models/MLP.pkl'...")
        with open('models/MLP.pkl', 'wb') as f:
            pickle.dump(MLP, f)

        logging.info("Model training completed successfully.")
        return MLP, xtest_scaled, ytest

    except Exception as e:
        logging.error("Error occurred during MLP model training", exc_info=True)
        raise
