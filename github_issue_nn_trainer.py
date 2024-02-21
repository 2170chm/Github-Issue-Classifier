import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
import time

def load_data(representation):
    X_train = np.load(f'text_representation/multiclass/X_train_{representation}.npy')
    y_train = np.load(f'text_representation/multiclass/y_train.npy')
    X_test = np.load(f'text_representation/multiclass/X_test_{representation}.npy')
    y_test = np.load(f'text_representation/multiclass/y_test.npy')
    return X_train, y_train, X_test, y_test

def perform_grid_search(model, X_train, y_train):
    param_grid = {
        # 'hidden_layer_sizes': [(50,), (100,)],
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        # 'alpha': [0.01, 0.1],
        'alpha': [0.001, 0.01, 0.1],
        'max_iter': [200, 500],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    representation = 'embeddings'
    
    if not os.path.exists('best_model'):
        os.makedirs('best_model')
    
    X_train, y_train, X_test, y_test = load_data(representation)
    
    print("Performing grid search to tune the neural network on word embeddings")
    start = time.time()
    
    model = MLPClassifier()
    best_nn_model = perform_grid_search(model, X_train, y_train)
    
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    evaluate_model(best_nn_model, X_test, y_test)
    
    model_path = 'best_model/nn_embeddings_tuned.joblib'
    joblib.dump(best_nn_model, model_path)

