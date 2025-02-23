import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainingArguments, Trainer

from Task2.CM.cm_model import CMModel
from Task2.CM.data import prepare_data


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average='macro'),
        "recall": recall_score(labels, predictions, average='macro'),
        "f1": f1_score(labels, predictions, average='macro'),
    }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def create_trainer(model, dataset, processor):
    args = TrainingArguments(
        use_cpu=True,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        weight_decay=0.01,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor
    )
    return trainer

def evaluate(trainer, dataset):
    test_results = trainer.evaluate(dataset['test'])
    print(f'Loss: {test_results["eval_loss"]}\n'
          f'Precision: {test_results["eval_precision"]}\n'
          f'Recall: {test_results["eval_recall"]}\n'
          f'F1: {test_results["eval_f1"]}\n')

def run_train_and_eval(model, savedir):
    dataset = prepare_data('/Users/vika/Desktop/dataset')

    cm_model = CMModel(model)
    processor, model = cm_model.ger_model_and_processor()

    trainer = create_trainer(model, dataset, processor)
    trainer.train()

    os.makedirs(savedir, exist_ok=True)
    trainer.save_model(f"{savedir}/cm_model")

    evaluate(trainer, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CM train')
    parser.add_argument('--m', type=str, default='google/vit-base-patch16-224', help='Model type')
    parser.add_argument('--s', type=str, default='../models', help='Save directory')
    args = parser.parse_args()
    run_train_and_eval(model=args.m, savedir=args.s)