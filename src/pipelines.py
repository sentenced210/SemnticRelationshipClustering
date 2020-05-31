from src.embedders import BertEmbedder
from src.tokenizers import TokenizerBert


class SentenceEmbedder(object):
    def __init__(self, tokenizer_data_path, embedder_data_path):
        self.embedder = BertEmbedder(embedder_data_path)  # /data/Embedders/BertModel/
        self.tokenizer = TokenizerBert(tokenizer_data_path)  # /data/Tokenizers/BertTokenizer/

    def __call__(self, sentence):
        x = self.tokenizer.tokenize(sentence)
        x = self.embedder.embed(x)
        return x


if __name__ == '__main__':
    e = SentenceEmbedder(embedder_data_path='../data/Embedders/BertModel/',
                         tokenizer_data_path='../data/Tokenizers/BertTokenizer/')
    print(e("My name is Amir"))
