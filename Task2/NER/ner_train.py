import os
import argparse
import numpy as np

from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score

from Task2.NER.data import get_dataset_dict, tokenize
from Task2.NER.ner_model import id2label, NERModel


def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, true_labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, true_labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

def create_trainer(model, tokenized_datasets, tokenizer):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    args = TrainingArguments(
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    return trainer

def evaluate(trainer, tokenized_datasets):
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f'Loss: {test_results["eval_loss"]}\n'
          f'Precision: {test_results["eval_precision"]}\n'
          f'Recall: {test_results["eval_recall"]}\n'
          f'F1: {test_results["eval_f1"]}\n')


def run_train_and_eval(model, savedir):
    ner_model = NERModel(model)
    tokenizer, model = ner_model.ger_model_and_tokenizer()

    dataset_dict = get_dataset_dict()
    tokenized_dataset_dict = tokenize(dataset_dict, tokenizer)

    trainer = create_trainer(model, tokenized_dataset_dict, tokenizer)
    trainer.train()

    os.makedirs(savedir, exist_ok=True)
    trainer.save_model(f"{savedir}/ner_model")

    evaluate(trainer, tokenized_dataset_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='NER train')
    parser.add_argument('--m', type=str, default='distilbert-base-uncased', help='Model type')
    parser.add_argument('--s', type=str, default='../models', help='Save directory')
    args = parser.parse_args()
    run_train_and_eval(model=args.m, savedir=args.s)

