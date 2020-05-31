import torch
import torch.nn as nn

from transformers import BertModel
from laserembeddings import Laser


class Embedder(object):
    def embed(self, tokenized_text):
        pass


class BertEmbedder(Embedder):
    def __init__(self, data_path):
        super().__init__()

        self.model = BertModel.from_pretrained(data_path)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def embed(self, tokenized_text):
        if torch.cuda.is_available():
            tokenized_text = tokenized_text.to('cuda')

        with torch.no_grad():
            outputs = self.model(tokenized_text)
            encoded_layers = outputs[0]
            mean = torch.mean(encoded_layers, dim=1)
        if torch.cuda.is_available():
            mean = mean.cpu()
        return mean.numpy()


class LASEREmbedder(Embedder):
    def __init__(self, tokenizer_language):
        super().__init__()
        self.laser = Laser()
        self.tokenizer_language = tokenizer_language

    def embed(self, sentence):
        return self.laser.embed_sentences(sentence, self.tokenizer_language)[0]


