import unittest
from src.data_loader import load_data, split_data
from src.model import train, evaluate

class TestModel(unittest.TestCase):

    def test_load_data(self):
        data = load_data()
        self.assertEqual(data.shape, (150, 5))

    def test_model_training(self):
        data = load_data()
        X_train, X_test, y_train, y_test = split_data(data)
        model = train(X_train, y_train)
        accuracy = evaluate(model, X_test, y_test)
        self.assertGreater(accuracy, 0.9)

if __name__ == '__main__':
    unittest.main()
