import argparse

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from Task2.NER.data import process_sentence


def load_model(model_type, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
    return pipe


def run_inference(model_type, load_directory):
    pipe = load_model(model_type, f'{load_directory}/ner_model')
    text = input('Enter your input: ')
    text = process_sentence(text)
    results = pipe(text)
    for result in results:
        if len(result) != 0:
            print(result[0]['word'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='NER inference')
    parser.add_argument('--m', type=str, default='distilbert-base-uncased', help='Model type')
    parser.add_argument('--l', type=str, default='../models', help='Model to load')
    args = parser.parse_args()
    run_inference(model_type=args.m, load_directory=args.l)
