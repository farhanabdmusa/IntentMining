from transformers import AutoTokenizer, AutoModel
import torch

class IndobertEmbedding:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loadModel()

    def loadModel(self):
        if self.model is None:
            print("Loading HuggingFace IndoBERT model...")
            model_name = "indobenchmark/indobert-base-p1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print("Model Loaded.")

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().tolist()

    def getEmbeddings(self, data):
        vector = []

        for i in data:
            samples = i
            features = self.encode(samples)
            vector += features

        return vector
