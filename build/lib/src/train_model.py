import argparse
from data_loader import load_data, split_data
from model import train, evaluate, plot_tree_model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Decision Tree model on the Iris dataset.")
    args = parser.parse_args()
    
    # Load and split the data
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train the model
    model = train(X_train, y_train)
    
    # Evaluate the model
    evaluate(model, X_test, y_test)
    
    # Plot the tree
    feature_names = data.columns[:-1].tolist()
    class_names = ['setosa', 'versicolor', 'virginica']
    plot_tree_model(model, feature_names, class_names)

if __name__ == "__main__":
    main()
