import argparse

from PIL import Image

from CM.cm_inference import load_cm_model
from NER.data import process_sentence
from NER.ner_inference import load_ner_model


def main(ner_model, cm_model, path_to_image, text):
    """
    Combined NER and classification pipeline to determine truth of text statement about image
    """
    ner_model = load_ner_model('distilbert-base-uncased', ner_model)
    cm_model = load_cm_model('google/vit-base-patch16-224', cm_model)

    image = Image.open(path_to_image)
    cm_label = cm_model(image)[0]['label']

    text = process_sentence(text)
    results = ner_model(text)
    for result in results:
        if len(result) != 0:
            ner_label = result[0]['word']
            break
    print(f'Statement is {cm_label.lower() == ner_label.lower()}')
    return cm_label.lower() == ner_label.lower()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NER+IM')
    parser.add_argument('--ner', type=str, default='models/ner_model', help='NER model path')
    parser.add_argument('--cm', type=str, default='models/cm_model', help='CM model path')
    parser.add_argument('--im', type=str, help='Path to image to run inference')
    parser.add_argument('--txt', type=str, help='Text to run inference')
    args = parser.parse_args()
    main(ner_model=args.ner, cm_model=args.cm, path_to_image=args.im, text=args.txt)

