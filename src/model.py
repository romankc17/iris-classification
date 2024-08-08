from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def train(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy

def plot_tree_model(model, feature_names, class_names):
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()
