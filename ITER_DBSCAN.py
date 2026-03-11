import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances,homogeneity_score,completeness_score,normalized_mutual_info_score,adjusted_mutual_info_score,adjusted_rand_score,silhouette_score,davies_bouldin_score,calinski_harabasz_score,accuracy_score,precision_score,recall_score, f1_score
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sentenceEmbedding import SentenceEmbedding
from IndobertEmbedding import IndobertEmbedding
from IndoSBERTEmbedding import IndoSBERTEmbedding
import re


class ITER_DBSCAN(DBSCAN):
    """
    ITER-DBSCAN Implementation - Iteratively adapt dbscan parameters for unbalanced data (text) clustering
    The change of core parameters of DBSCAN i.e. distance and minimum samples parameters are changed smoothly to
    find high to low density clusters. At each iteration distance parameter is increased by 0.01 and minimum samples
    are decreased by 1. The algorithm uses cosine distance for cluster creation
    :params
    :initial_distance: initial distance for initial cluster creation (default: 0.10)
    :initial_minimum_samples: initial minimum sample count for initial cluster creation (default: 20)
    :delta_distance: change in distance parameter at each iteration(default: 0.01)
    :delta_minimum_samples: change in minimum sample parameter (of DBSCAN) at each iteration(default: 0.01)
    :max_iteration : maximum number of iteration the DBSCAN algorithm will run for cluster creation(default: 5)
    :threshold: threshold parameter controls the size of the cluster, any cluster contains more than threshold parameter
                will be discarded. (default: 300)
    :features: default values is None, the algorithm expects a list of short texts. In case the representation is
                pre-computed for text or data sources (pass featyres values as "precomputed").
    :embedding_model: ITER-DBSCAN or IndoBERT
    :metric: euclidean, precomputed, etc
    """

    def __init__(self, initial_distance=0.10, initial_minimum_samples=20, delta_distance=0.01, delta_minimum_samples=1,
                 max_iteration=5, threshold=300, features=None, embedding_model="ITER-DBSCAN", metric="precomputed"
                 ):

        self.initial_distance = initial_distance
        self.initial_minimum_samples = initial_minimum_samples
        self.delta_distance = delta_distance
        self.delta_minimum_samples = delta_minimum_samples
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.features = features
        self.embedding_model = embedding_model
        self.metric = metric
        self.labels_ = None

    def preprocess_data(self, features):
        self.features = features
        clean_text_array = []
        for feature in features:
            text = re.sub(r'[^\w\s]', ' ', feature)
            text = text.lower()
            tokens = text.split(' ')
            tokens = [token.strip() for token in tokens if len(token.strip()) > 0]
            text = ' '.join(tokens)
            text = text.strip()
            clean_text_array.append(text)

        return clean_text_array

    def compute(self, data):
        if not (type(data) is np.ndarray or type(data) is list):
            raise Exception("Please pass a list of string or a list of feature vectors.")

        if type(data[0]) is str:
            #data = self.preprocess_data(data)
            if self.features != 'precomputed':
                print(f"Create vector embedding for {self.embedding_model}...")
                if self.embedding_model == "ITER-DBSCAN":
                    embedding_model = SentenceEmbedding()
                    data = embedding_model.getEmbeddings(data)
                elif self.embedding_model == "IndoBERT":
                    embedding_model = IndobertEmbedding()
                    data = embedding_model.getEmbeddings(data)
                elif self.embedding_model == "IndoSBERT":
                    embedding_model = IndoSBERTEmbedding()
                    data = embedding_model.getEmbeddings(data)
                else:
                    raise Exception("Invalid embedding_model")
                print("Vector embedding created!")


        df = pd.DataFrame(index=range(len(data)), columns=['features', 'labels'])
        df['features'] = data
        df['labels'] = [-1] * len(data)
        cluster_id = 0
        for i in range(self.max_iteration):
            # print(f"iterasi ke-{i}")
            # print(f"initial_distance: {self.initial_distance}")
            # print(f"initial_minimum_samples: {self.initial_minimum_samples}")
            features = np.array(df.loc[df.labels == -1]['features'].values.tolist())
            if self.metric == "precomputed":
                distance_matrix = pairwise_distances(features, metric='cosine')
            else :
                distance_matrix = features

            if 5 > len(features): break
            cluster_labels = DBSCAN(eps=self.initial_distance, 
                                    min_samples=self.initial_minimum_samples,
                                    metric=self.metric)
            labels = cluster_labels.fit_predict(distance_matrix)
            cluster_labels = [str(c) for c in labels]
            # print(f"cluster_labels:\n{cluster_labels}")
            label_freq = Counter(cluster_labels)
            # print(f"label_freq:\n{label_freq}")
            label_set = cluster_labels
            new_label_set = [-1 if label_freq[l] > self.threshold or l == '-1' else int(l) + cluster_id for l in
                             label_set]
            # print(f"new_label_set:\n{new_label_set}")

            cluster_values = [k for k in new_label_set if k != -1]
            # print(f"cluster_values:\n{cluster_values}")

            if len(cluster_values) > 0:
                min_cluster_id = min(cluster_id, min(cluster_values))
            else:
                min_cluster_id = cluster_id
            max_cluster_id = min_cluster_id + len(list(set(cluster_values)))

            unique_cluster_ids = list(set(cluster_values))
            # print(f"unique_cluster_ids:\n{unique_cluster_ids}")
            id_mapper = dict()
            for i in range(len(unique_cluster_ids)):
                id_mapper[unique_cluster_ids[i]] = min_cluster_id
                min_cluster_id += 1

            # print(f"id_mapper:\n{id_mapper}")

            new_label_set = [-1 if l == -1 else id_mapper[l] for l in new_label_set]
            # print(f"new_label_set:\n{new_label_set}")
            cluster_id = max_cluster_id
            # print(f"cluster_id:\n{cluster_id}")
            df.loc[df['labels'] == -1, 'labels'] = new_label_set
            self.initial_distance += self.delta_distance
            self.initial_minimum_samples -= self.delta_minimum_samples
            # print("="*10)

            if self.initial_minimum_samples == 2:
                break

        self.labels_ = df['labels'].values.tolist()

    def fit_predict(self, X):
        self.compute(X)
        return self.labels_

    def fit(self, X):
        self.compute(X)

    def generate_labels(self, df, target_column):
        """calculate cluster purity
        """
        df['representative_label'] = ['None'] * len(df)
        total = 0
        noise_count = 0
        purities = []
        for cluster_id in df.cluster_ids.unique():
            if cluster_id == 'None': continue
            from collections import Counter
            tmp_df = df.loc[df.cluster_ids == cluster_id][target_column].values.tolist()
            counts = Counter(tmp_df)
            intent = None
            cur_value = 0
            for key, value in counts.items():
                if value > cur_value:
                    cur_value = value
                    intent = key
            if len(tmp_df) == 0: continue
            purity = round(cur_value / len(tmp_df), 2)
            purities.append(purity)
            total += purity
            if purity >= 0.5:
                df.loc[df.cluster_ids == cluster_id, 'representative_label'] = intent
            else:
                noise_count += 1
        
        return noise_count, df
    
    def label_propagation(self, df):
        """propagate labels to unlabelled samples
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LogisticRegression
        X = np.array(df.loc[df.representative_label != 'None']['features'].values.tolist())
        labels = df.loc[df.representative_label != 'None']['representative_label'].values.tolist()
        le = LabelEncoder()
        y = le.fit_transform(labels)
        clf = LogisticRegression(class_weight='balanced', C=0.8, solver='newton-cg')
        clf.fit(X, y)
        feat = np.array(df['features'].values.tolist())
        y_pred = clf.predict(feat)
        labels = [le.classes_[i] for i in y_pred]
        df['predictedIntent'] = labels
        return df
    
    def extract_feature(self, text_column, df):
        """
        extract feature representation of short text using Universenal sentence encoder
        :return:
        """
        data = df[text_column].values.tolist()
        if self.embedding_model == "ITER-DBSCAN":
            sentenceEmbedding = SentenceEmbedding()
            feature = sentenceEmbedding.getEmbeddings(data)
        elif self.embedding_model == "IndoBERT":
            self.embedding_model = IndobertEmbedding()
            feature = self.embedding_model.getEmbeddings(data)
        elif self.embedding_model == "IndoSBERT":
            self.embedding_model = IndoSBERTEmbedding()
            feature = self.embedding_model.getEmbeddings(data)
        else:
            raise Exception("Invalid embedding_model")
        df['features'] = feature
        return df

    def compute_evaluate(self, filetype, filename, text_column, target_column):
        if filetype not in ['csv', 'xlsx']:
            raise Exception("Only supports csv and excel file.")
        try:
            if filetype == 'csv':
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
        except:
            raise Exception("Failed to load file!!")
        
        df = self.extract_feature(text_column=text_column, df=df)

        df['labels'] = [-1] * len(df[text_column])
        cluster_id = 0
        param_results = []
        for i in range(self.max_iteration):
            # print(f"iterasi ke-{i}")
            # print(f"initial_distance: {self.initial_distance}")
            # print(f"initial_minimum_samples: {self.initial_minimum_samples}")
            features = np.array(df.loc[df.labels == -1]['features'].values.tolist())
            if self.metric == "precomputed":
                distance_matrix = pairwise_distances(features, metric='cosine')
            else :
                distance_matrix = features

            if 5 > len(features): break
            cluster_labels = DBSCAN(eps=self.initial_distance, 
                                    min_samples=self.initial_minimum_samples,
                                    metric=self.metric)
            labels = cluster_labels.fit_predict(distance_matrix)
            cluster_labels = [str(c) for c in labels]
            # print(f"cluster_labels:\n{cluster_labels}")
            label_freq = Counter(cluster_labels)
            # print(f"label_freq:\n{label_freq}")
            label_set = cluster_labels
            new_label_set = [-1 if label_freq[l] > self.threshold or l == '-1' else int(l) + cluster_id for l in
                             label_set]
            # print(f"new_label_set:\n{new_label_set}")

            cluster_values = [k for k in new_label_set if k != -1]
            # print(f"cluster_values:\n{cluster_values}")

            if len(cluster_values) > 0:
                min_cluster_id = min(cluster_id, min(cluster_values))
            else:
                min_cluster_id = cluster_id
            max_cluster_id = min_cluster_id + len(list(set(cluster_values)))

            unique_cluster_ids = list(set(cluster_values))
            # print(f"unique_cluster_ids:\n{unique_cluster_ids}")
            id_mapper = dict()
            for i in range(len(unique_cluster_ids)):
                id_mapper[unique_cluster_ids[i]] = min_cluster_id
                min_cluster_id += 1

            # print(f"id_mapper:\n{id_mapper}")

            new_label_set = [-1 if l == -1 else id_mapper[l] for l in new_label_set]
            # print(f"new_label_set:\n{new_label_set}")
            cluster_id = max_cluster_id
            # print(f"cluster_id:\n{cluster_id}")
            df.loc[df['labels'] == -1, 'labels'] = new_label_set
            self.initial_distance += self.delta_distance
            self.initial_minimum_samples -= self.delta_minimum_samples
            # print("="*10)

            # Start Evaluate
            cluster_labels = ['None' if c == -1 else c for c in df['labels'].values.tolist()]
            df['cluster_ids'] = cluster_labels
            noise_count, df = self.generate_labels(df=df, target_column=target_column)

            if 2 > len(df[df['representative_label'] != 'None']['representative_label'].unique()):
                continue
            
            df = self.label_propagation(df)

            # print(f"cluster_ids:\n{df.cluster_ids.value_counts()}")
            per_labelled = round(len(df.loc[df.cluster_ids != 'None']) / len(df) * 100, 2)
            # print(f"per_labelled:\n{per_labelled}")
            num_clusters = len(list(set(df.loc[df.cluster_ids != 'None']['cluster_ids'].values.tolist())))
            # print(f"num_clusters:\n{num_clusters}")
            true_intent = df[target_column].values.tolist()
            # print(f"true_intent:\n{true_intent}")
            predicted_intent = df['predictedIntent'].values.tolist()
            # print(f"predicted_intent:\n{predicted_intent}")
            h_score = round(homogeneity_score(true_intent, predicted_intent), 2)
            c_score = round(completeness_score(true_intent, predicted_intent), 2)
            nmf = round(normalized_mutual_info_score(true_intent, predicted_intent), 2)
            amf = round(adjusted_mutual_info_score(true_intent, predicted_intent), 2)
            ars = round(adjusted_rand_score(true_intent, predicted_intent), 2)

            le = LabelEncoder()

            le = le.fit(df[target_column].values.tolist())

            true = le.transform(df[target_column].values.tolist())
            pred = le.transform(df['predictedIntent'].values.tolist())
            silhouette = silhouette_score(df["features"].values.tolist(), pred)
            davies_bouldin = davies_bouldin_score(df["features"].values.tolist(), pred)
            calinski = calinski_harabasz_score(df["features"].values.tolist(), pred)
            accuracy = accuracy_score(true, pred)
            precision = precision_score(true, pred, average='weighted')
            recall = recall_score(true, pred, average='weighted')
            f1 = f1_score(true, pred, average='weighted')

            param = {}
            param['initial_distance'] = self.initial_distance
            param['initial_minimum_samples'] = self.initial_minimum_samples
            param['delta_distance'] = self.delta_distance
            param['delta_minimum_samples'] = self.delta_minimum_samples
            param['max_iteration'] = self.max_iteration
            param['threshold'] = self.threshold
            param['features'] = self.features
            param['embedding_model'] = f"{self.embedding_model}"
            param['metric'] = self.metric
            param['iteration'] = i
            param['percentage_labelled'] = per_labelled
            param['clusters'] = num_clusters
            param['noisy_clusters'] = noise_count
            param['homogeneity_score'] = h_score
            param['completeness_score'] = c_score
            param['normalized_mutual_info_score'] = nmf
            param['adjusted_mutual_info_score'] = amf
            param['adjusted_rand_score'] = ars
            param['silhouette_score'] = round(silhouette, 3)
            param['davies_bouldin'] = round(davies_bouldin, 3)
            param['calinski_harabasz'] = round(calinski, 3)
            param['accuracy'] = round(accuracy, 3) * 100.0
            param['precision'] = round(precision, 3) * 100.0
            param['recall'] = round(recall, 3) * 100.0
            param['f1'] = round(f1, 3) * 100.0
            param['accuracy'] = accuracy

            param['intents'] = len(
                df.loc[df['representative_label'] != 'None']['representative_label'].value_counts())
            param_results.append(param)
            # End Evaluate

            if self.initial_minimum_samples == 2:
                break

        self.labels_ = df['labels'].values.tolist()
        return param_results
    
    def fit_predict_evaluate(self, filetype, filename, text_column, target_column):
        evaluate_result = self.compute_evaluate(filetype, filename, text_column, target_column)
        return self.labels_, evaluate_result