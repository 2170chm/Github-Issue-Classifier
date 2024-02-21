import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_test_data(representation):
    X_test = np.load(f'text_representation/multiclass/X_test_{representation}.npy')
    y_test = np.load(f'text_representation/multiclass/y_test.npy')
    
    return X_test, y_test

def load_models(representation, model_type, one_vs_all):
    if one_vs_all:
        model_paths = [f'models/{model_type}_{representation}_one_vs_all_class_{i}.joblib' for i in range(3)]
    else:
        model_paths = [f'models/{model_type}_{representation}_multiclass.joblib']
    
    models = [joblib.load(path) for path in model_paths]
    
    if one_vs_all:
        return models
    else:
        return models[0]

def vote_predictions(models, X_test):
    num_samples = X_test.shape[0]
    
    voted_predictions = []

    for i in range(num_samples):
        sample_predictions = [model.predict(X_test[i].reshape(1, -1)) for model in models]
        sample_predictions = np.array(sample_predictions).flatten()
        
        voted_prediction = np.argmax(np.bincount(sample_predictions))
        voted_predictions.append(voted_prediction)

    voted_predictions = np.array(voted_predictions)
    
    return voted_predictions

def evaluate_model(model, X_test, y_test, model_name, representation):
    print(f"Metrics for model {model_name} with {representation} representation")
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    representations = ['tfidf', 'lda', 'embeddings']
    model_types = ['rf', 'nb', 'nn', 'svm']

    for representation in representations:
        X_test, y_test = load_test_data(representation)
        
        for model_type in model_types:
            if model_type == 'svm':
                svm_models = load_models(representation, 'svm', True)
                voted_predictions = vote_predictions(svm_models, X_test)
                print(f"Metrics for SVM with {representation} representation (one-vs-all):")
                print(f"Accuracy: {accuracy_score(y_test, voted_predictions)}")
                print(f"F1 Score: {f1_score(y_test, voted_predictions, average='weighted')}")
                print(classification_report(y_test, voted_predictions))
            else:
                model = load_models(representation, model_type, False)
                evaluate_model(model, X_test, y_test, model_type, representation)
