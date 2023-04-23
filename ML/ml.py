from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class myModel:
    def __init__(self, model_name):
        self.classifier = {
            "LR": LogisticRegression(penalty='l2', max_iter=10000),
            "RF": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "MLP": MLPClassifier(hidden_layer_sizes=(512), max_iter=10000, random_state=1),
            "SVC": SVC(kernel='rbf', probability=True, gamma=10, C=10)
        }
        
        assert model_name in self.classifier.keys(), "There is no model for you"
        self.model_name = model_name
        self.model = self.classifier[model_name]
    
    def train(self, data):
        x_train, y_train = data
        self.model.fit(x_train, y_train)
    
    def test(self, data):
        x_test, y_test = data
        y_pred = self.model.predict(x_test)
        results = { 
            "model_name": self.model_name,
            "accuracy_score": metrics.accuracy_score(y_test, y_pred),
            "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
            "f1_score": metrics.f1_score(y_test, y_pred, average="macro"),
            "recall_score": metrics.recall_score(y_test, y_pred, average="macro"),
        }
        return results
    
    def predict(self, X):
        return self.model.predict(X)
    

