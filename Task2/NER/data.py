import ast

from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from Task2.NER.ner_model import NERModel
import string
import contractions

import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


MODEL_NAME = 'distilbert-base-uncased'


def process_sentence(sentence):
    lemmatizer = WordNetLemmatizer()

    sentence = contractions.fix(sentence)
    tokens = word_tokenize(sentence)
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token.lower() for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def get_dataset_dict(csv_path="Task2/NER/ner_dataset.csv", split_rate=0.2):
    def create_dataset(df):
        created_df = df.rename(columns={'Unnamed: 0': 'id'})
        created_df['words'] = created_df['words'].apply(lambda x: ast.literal_eval(x))
        created_df['ner_tags'] = created_df['ner_tags'].apply(lambda x: ast.literal_eval(x))
        return Dataset.from_pandas(created_df)

    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=split_rate)
    train_df, val_df = train_test_split(train_df, test_size=split_rate)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return DatasetDict({"train": create_dataset(train_df),
                        "validation": create_dataset(val_df),
                        "test": create_dataset(test_df)})

def tokenize(data, tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["words"], truncation=True, padding=True, is_split_into_words=True)
        all_labels = []

        for index, ner_tags in enumerate(examples["ner_tags"]):
            word_id_list = tokenized.word_ids(batch_index=index)
            prev_word_idx = None
            aligned_labels = []
            for word_id in word_id_list:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != prev_word_idx:
                    aligned_labels.append(ner_tags[word_id])
                else:
                    aligned_labels.append(-100)
                prev_word_idx = word_id
            all_labels.append(aligned_labels)

        tokenized["labels"] = all_labels
        return tokenized
    return data.map(tokenize_and_align_labels,
                    batched=True,
                    remove_columns=data["train"].column_names
                )

if __name__ == "__main__":
    nerModel = NERModel(MODEL_NAME)
    tokenizer, _ = nerModel.ger_model_and_tokenizer()
    dataset_dict = get_dataset_dict()
    tokenized_dataset_dict = tokenize(dataset_dict, tokenizer)
    print(tokenized_dataset_dict)
