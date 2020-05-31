import torch
from transformers import BertModel, BertTokenizer


class Tokenizer(object):
    def tokenize(self, tokenized_text):
        pass


class TokenizerBert(Tokenizer):
    def __init__(self, data_path):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(data_path)

    def tokenize(self, sentence):
        sentence = "[CLS] " + sentence + "[SEP]"
        tokenized_text = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

        return tokens_tensor

