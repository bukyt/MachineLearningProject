import os
from PIL import Image
import pytesseract
import numpy as np
import torch
import nltk
from estnltk import Text
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import logging


"""
Summarize text võtab ette argumentideks kokkuvõtte pikkuse (lausete arv)
ja teksti mida kokku võtta
"""
def summarize_text(text, summary_length):

    #tekst lausete listiks
    sentences = split_text_into_sentences(text)

    #igast lausest embedding
    embeddings = generate_sentence_embeddings(sentences)


    # klasterdamine, tagastab listi kus on klastri numbrid samas järjekorras nagu laused ette anti, teeme nii mitu klastrit, kui mitut lauset kokkuvõttesse tahame
    labels = k_means_clustering(embeddings, summary_length)

    for i in range(len(sentences)):
        print ("Klastrisse "+ str (labels[i]) + " määrati lause "+sentences[i])

    # kokkuvõtte tegmiseks võtame klastrite kõige esinduslikumad laused iga klastri kõige esinduslikum võiks olla see, mis on kõige rohkem keskel (seda kasutavad ka teised summarizerid)
    #todo - vaadata üle, kas see klastri kõige keskmise leidja üldse teeb mida peaks
    centroids = []
    for cluster in range(summary_length):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids.append(centroid)

    summary = []
    for centroid in centroids:
        distances = [np.linalg.norm(embedding - centroid) for embedding in embeddings]
        closest_index = np.argmin(distances)
        closest_sentence = sentences[closest_index]
        summary.append(closest_sentence)

    return ' '.join(summary)
"""
Teksti lauseteks eraldamine
"""
def split_text_into_sentences(text):
    t = Text(text)
    t.tag_layer(['sentences'])
    sentences = t['sentences']
    sentence_list = [' '.join([word.text for word in sentence.words]) for sentence in sentences]
    #print(sentence_list)
    return sentence_list


"""
lausete embeddingute leidmine Estbert abil
lähenemise kuidas saada word embeddings põhjal lause omad võtsime siit https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
"""
def get_sentence_embeddings(sentences):

    #et väljundist kõiksugu hoiatused ära saada
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    #kasutame EstBERTi
    model_name = 'tartuNLP/EstBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        #computes the mean of the last hidden state along the sequence dimension (dim=1) to get a single vector representing the sentence. This vector is detached from the computation graph and converted to a NumPy array.
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())


    return embeddings
# 1) kõige sarnasemte lausete valimine


def get_summary(embeddings, sentences, num_sentences):
  # leiame lausete embeddingute keskmise ehk embeddingu, mis võiks vastata kogu tekstile
  text_embedding = np.mean(embeddings, axis=0)
  #käime läbi kõikide lausete embeddingute listi, leiame iga lause embeddingu ja kogu teksti embeddingu vahelise cosine similarity
  #loomuliku keele töötluses embeddingute puhul on just cosine similarity kõige levinum/parem viis kahe sõnavtori vahelise kauguse/sarnasuse leidmiseks https://medium.com/@techclaw/cosine-similarity-between-two-arrays-for-word-embeddings-c8c1c98811b
  similarities = [cosine_similarity(emb, text_embedding) for emb in embeddings]

  # järjestame laused sarnasuse alusel ja valime esimesed num_sentences ehk kõige sarnasemad
  ranked_sentences = [sent for _, sent in sorted(zip(similarities, sentences), key=lambda x: x[0], reverse=True)]
  summary = ' \n'.join(ranked_sentences[:num_sentences])
  return summary

def summarize_text_similarity(text, num_sentences):

    sentences = split_text_into_sentences(text)
    embeddings = get_sentence_embeddings(sentences)
    summary = get_summary(embeddings, sentences, num_sentences)
    return summary


# 2) klasterdamine

def k_means_clustering(embeddings, num_clusters):
    embeddings = np.vstack(embeddings)  # Muudame embeddinute loendi NumPy massiiviks
    #jagame embeddingut num_clusters arvu klastritesse
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(embeddings)
    #print(kmeans.labels_)  # Prindib välja, millisesse klastrisse laused kuuluvad
    return kmeans.labels_, kmeans.cluster_centers_

def summarize_cluster(text, summary_length):

    # Tekst lausete listiks
    sentences = split_text_into_sentences(text)

    # Iga lause embedding
    embeddings = get_sentence_embeddings(sentences)

    # Klasterdamine, tagastab labels listi, kus on igale lausele vastav klastri number samas järjekorras nagu laused ette anti, teeme nii mitu klastrit, kui mitut lauset kokkuvõttesse tahame, cluster_centers on iga klastri keskpunkt
    labels, cluster_centers = k_means_clustering(embeddings, summary_length)

    #print(cluster_centers)

#et vaadata, kuidas algoritm klastritesse jagab, kommenteeri järgmised read välja
#    for i in range(len(sentences)):
#        print("Klastrisse " + str(labels[i]) + " määrati lause " + sentences[i])

    # Kokkuvõtte tegemiseks võtame klastrite kõige esinduslikumad laused, iga klastri kõige esinduslikum võiks olla see, mis on kõige rohkem keskel (seda kasutavad ka teised summarizerid)
    summary = []
    for i in range(summary_length):
        #võtame välja need embeddingud mis kuuluvad samasse klastrisse
        cluster_indices = np.where(labels == i)[0]
        # paneme vastava klastri embeddingud ühte numpy arraysse
        cluster_embeddings = np.vstack([embeddings[j] for j in cluster_indices])
        #leiame selle klastri keskpunkti, keskpunktid tagastas k_means_clustering
        centroid = cluster_centers[i]
        #leiame iga lause embeddingu kauguse klastris keskpunktist
        distances = [np.linalg.norm(embedding - centroid) for embedding in cluster_embeddings]
        #leiame lähima lause embeddingu keskpunktile
        closest_index = cluster_indices[np.argmin(distances)]
        #leiame sellele vastava lause
        closest_sentence = sentences[closest_index]
        summary.append(closest_sentence)

    return ' \n'.join(summary)

#põhiline funktsioon - teeb kokkuvõtte tekstist
def summarize_text(text, summary_length):

    #tekst lausete listiks
    sentences = split_text_into_sentences(text)

    #igast lausest embedding
    embeddings = generate_sentence_embeddings(sentences)


    # klasterdamine, tagastab listi kus on klastri numbrid samas järjekorras nagu laused ette anti, teeme nii mitu klastrit, kui mitut lauset kokkuvõttesse tahame
    labels = k_means_clustering(embeddings, summary_length)

    #for i in range(len(sentences)):
    #    print ("Klastrisse "+ str (labels[i]) + " määrati lause "+sentences[i])

    # kokkuvõtte tegmiseks võtame klastrite kõige esinduslikumad laused iga klastri kõige esinduslikum võiks olla see, mis on kõige rohkem keskel (seda kasutavad ka teised summarizerid)
    #todo - vaadata üle, kas see klastri kõige keskmise leidja üldse teeb mida peaks
    centroids = []
    for cluster in range(summary_length):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids.append(centroid)

    summary = []
    for centroid in centroids:
        distances = [np.linalg.norm(embedding - centroid) for embedding in embeddings]
        closest_index = np.argmin(distances)
        closest_sentence = sentences[closest_index]
        summary.append(closest_sentence)

    return ' '.join(summary)

#võtab teksti kokku
def image_to_summary(path_to_img, number_of_sentences, algorithm):
    #tuvastame pildilt teksti
    extracted_information = pytesseract.image_to_string(Image.open(path_to_img), lang="est")
    if algorithm=="cluster":
        return summarize_cluster(extracted_information, number_of_sentences)
    if algorithm=="top_sentences":
        return summarize_text_similarity(extracted_information, number_of_sentences)

"""
väike setup, et kõik töötaks enne käivitamist
"""
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseractData = "tessdata"
"""
Sisendite küsimine
"""
path_to_image=input("Sisesta faili nimi: ")
num_sentences_in_summary=(int)(input("Sisesta mitu lauset kokkuvõttes on: "))
algorithm=(int)(input("Vali algoritm:\n1. Sarnasus\n2. Klasterdamine\n"))
extracted_information = pytesseract.image_to_string(path_to_image,lang="est",config="--psm 1")

"""
prindib leitud teksti kokkuvõtte
"""
algoritmid=["top_sentences","cluster"]
print(extracted_information)
result=image_to_summary(path_to_image, num_sentences_in_summary, algoritmid[algorithm-1] )
print(result)