import argparse

from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor


def load_cm_model(model_type, model_path):
    processor = AutoImageProcessor.from_pretrained(model_type)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    pipe = pipeline("image-classification", model=model, image_processor=processor)
    return pipe


def run_inference(model_type, load_directory, image_path):
    pipe = load_cm_model(model_type, f'{load_directory}/cm_model')
    results = pipe(image_path)
    result = results[0]['label']
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CM inference')
    parser.add_argument('--m', type=str, default='google/vit-base-patch16-224', help='Model type')
    parser.add_argument('--l', type=str, default='Task2/models', help='Model to load')
    parser.add_argument('--im', type=str, help='Path to image to run inference')
    args = parser.parse_args()
    run_inference(model_type=args.m, load_directory=args.l, image_path=args.im)
