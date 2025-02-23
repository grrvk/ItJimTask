from transformers import ViTForImageClassification, ViTImageProcessor


idx2label = {0: 'Cat', 1: 'Cow', 2: 'Dog', 3: 'Elephant', 4: 'Gorilla', 5: 'Hippo', 6: 'Lizard', 7: 'Monkey', 8: 'Mouse', 9: 'Panda', 10: 'Tiger', 11: 'Zebra'}
label2idx = {'Cat': 0, 'Cow': 1, 'Dog': 2, 'Elephant': 3, 'Gorilla': 4, 'Hippo': 5, 'Lizard': 6, 'Monkey': 7, 'Mouse': 8, 'Panda': 9, 'Tiger': 10, 'Zebra': 11}


class CMModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def ger_model_and_processor(self):
        model = ViTForImageClassification.from_pretrained(self.model_name,
                                                          id2label=idx2label,
                                                          label2id=label2idx,
                                                          num_labels=12,
                                                          ignore_mismatched_sizes=True)
        processor = ViTImageProcessor.from_pretrained(self.model_name, do_rescale = False, return_tensors = 'pt')
        return processor, model