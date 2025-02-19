import argparse

from src.data_process import get_data
from src.classifiers import MnistClassifier


def main(algorithm: str, full_report: bool = False, custom_input_path: str = None):
    train_dataloader, test_dataloader = get_data()
    classifier = MnistClassifier(algorithm)

    classifier.train(train_dataloader)
    accuracy, macro_f1 = classifier.test(test_dataloader, full_report)
    print(f'Accuracy {accuracy}, macro f1 {macro_f1}')

    if custom_input_path is not None:
        classifier.predict(custom_input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mnist classifier')
    parser.add_argument('--alg', type=str, help='Algorithm type')
    parser.add_argument('--input', type=str, default=None, help='Custom data path')
    parser.add_argument('--fr', action=argparse.BooleanOptionalAction, help='Get test full report')
    args = parser.parse_args()

    main(algorithm=args.alg, full_report=args.fr, custom_input_path=args.input)
