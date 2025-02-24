from transformers import AutoModelForTokenClassification, AutoTokenizer

# correlation between tags and labels to init the model
unique_labels = ["O", "B-AN", "I-AN"]
label2id = {k: v for v, k in enumerate(unique_labels)}
id2label = {v: k for v, k in enumerate(unique_labels)}

class NERModel:
    """
    Class of the NER model
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def ger_model_and_tokenizer(self):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name,
                                                               num_labels=len(id2label),
                                                               id2label=id2label,
                                                               label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer, model