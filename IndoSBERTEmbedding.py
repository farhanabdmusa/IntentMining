from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class IndoSBERTEmbedding():
    def __init__(self):
        self.model = None
        self.loadModel()

    def loadModel(self):
        if self.model is None:
            print('Loading Huggingface IndoSBERT model....')
            self.model = SentenceTransformer('denaya/indoSBERT-large')
            print('Model Loaded.')

    def encode(self, text):
        try:
            embeddings = self.model.encode(text)
            return embeddings
        except Exception as e:
            print(e)
            return
        

    def getEmbeddings(self, data):
        vector = []

        for text in tqdm(data):
            features = self.encode(text)
            vector.append(features)

        return vector