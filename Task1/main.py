import argparse

from src.data_process import load_mnist_dataset, prepare_custom_input, print_predictions, get_dataloader
from src.classifiers import MnistClassifier


def main(algorithm: str, custom_input_path: str = None):
    train_dataset, test_dataset = load_mnist_dataset()
    train_dataloader = get_dataloader(train_dataset, 64)
    test_dataloader = get_dataloader(test_dataset, 64)

    classifier = MnistClassifier(algorithm)

    classifier.train(train_dataloader)
    y_pred, y_test = classifier.test(test_dataloader)

    if custom_input_path is not None:
        image_paths, prepared_data = prepare_custom_input(custom_input_path)
        predictions = classifier.predict(prepared_data)
        print_predictions(image_paths, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Mnist classifier')
    parser.add_argument('--alg', type=str, help='Algorithm type')
    parser.add_argument('--input', type=str, default=None, help='Custom data path')
    args = parser.parse_args()
    main(algorithm=args.alg, custom_input_path=args.input)
