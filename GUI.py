import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Scrollbar
from PIL import Image, ImageTk
import pytesseract
import numpy as np
import torch
from estnltk import Text
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sentence_transformers import SentenceTransformer, models, InputExample, losses


def summarize_text_similarity(text, num_sentences):
    sentences = split_text_into_sentences(text)
    embeddings = get_sentence_embeddings(sentences)
    summary = get_summary(embeddings, sentences, num_sentences)
    return summary


def summarize_cluster(embeddings, summary_length, sentences):

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

    return ''.join(summary)


def image_to_summary(path_to_img, number_of_sentences, algorithm):
    extracted_information = pytesseract.image_to_string(Image.open(path_to_img), lang="est")
    if algorithm == "cluster":
        return extracted_information, summarize_cluster(extracted_information, number_of_sentences)
    if algorithm == "top_sentences":
        return extracted_information, summarize_text_similarity(extracted_information, number_of_sentences)


def split_text_into_sentences(text):
    t = Text(text)
    t.tag_layer(['sentences'])
    sentences = t['sentences']
    sentence_list = [' '.join([word.text for word in sentence.words]) for sentence in sentences]
    return sentence_list


def get_sentence_embeddings(sentences):
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    model_name = 'tartuNLP/EstBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    return embeddings

def get_summary(embeddings, num_sentences, sentences):
  # leiame lausete embeddingute keskmise ehk embeddingu, mis võiks vastata kogu tekstile
  text_embedding = np.mean(embeddings, axis=0)
  #käime läbi kõikide lausete embeddingute listi, leiame iga lause embeddingu ja kogu teksti embeddingu vahelise cosine similarity
  #loomuliku keele töötluses embeddingute puhul on just cosine similarity kõige levinum/parem viis kahe sõnavtori vahelise kauguse/sarnasuse leidmiseks https://medium.com/@techclaw/cosine-similarity-between-two-arrays-for-word-embeddings-c8c1c98811b
  similarities = [cosine_similarity(emb, text_embedding) for emb in embeddings]

  # järjestame laused sarnasuse alusel ja valime esimesed num_sentences ehk kõige sarnasemad
  ranked_sentences = [sent for _, sent in sorted(zip(similarities, sentences), key=lambda x: x[0], reverse=True)]
  summary = ''.join(ranked_sentences[:num_sentences])

  return summary


def k_means_clustering(embeddings, num_clusters):
    embeddings = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.labels_, kmeans.cluster_centers_

def get_sentence_embeddings_mean_pooling(sentences):

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

def get_sentence_embeddings_sentenceTransformer(sentences):

    #et väljundid liigset infot ei sisaldaks
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    #  laeme SentenceTransformer mudeli
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    embeddings=[]
    # leiame mudeliga lausete embeddingud
    for sentence in sentences:
        embeddings.append([model.encode(sentence)])


    return embeddings

def summarize_cluster_mean_pooling(text, summary_length):
    # Tekst lausete listiks
    sentences = split_text_into_sentences(text)
     # Iga lause embedding
    embeddings = get_sentence_embeddings_mean_pooling(sentences)
    return summarize_cluster(embeddings, summary_length, sentences)

def summarize_cluster_sentenceTransformer(text, summary_length):
    # Tekst lausete listiks
    sentences = split_text_into_sentences(text)
     # Iga lause embedding
    embeddings = get_sentence_embeddings_sentenceTransformer(sentences)
    return summarize_cluster(embeddings, summary_length, sentences)

def summarize_similarity_mean_pooling(text, summary_length):
    # Tekst lausete listiks
    sentences = split_text_into_sentences(text)

     # Iga lause embedding
    embeddings = get_sentence_embeddings_mean_pooling(sentences)

    return get_summary(embeddings, summary_length, sentences)


def summarize_similarity_sentenceTransformer(text, summary_length):
    # Tekst lausete listiks
    sentences = split_text_into_sentences(text)

     # Iga lause embedding
    embeddings = get_sentence_embeddings_sentenceTransformer(sentences)
    return get_summary(embeddings, summary_length, sentences)

def image_to_summary(path_to_img, number_of_sentences, algorithm, embeddings):
    #tuvastame pildilt teksti
    extracted_information = pytesseract.image_to_string(Image.open(path_to_img), lang="est")
    if algorithm=="cluster" and embeddings=="pooling":
        return extracted_information,summarize_cluster_mean_pooling(extracted_information, number_of_sentences)
    if algorithm=="cluster" and embeddings=="sentenceTransformer":
        return extracted_information,summarize_cluster_sentenceTransformer(extracted_information, number_of_sentences)
    if algorithm=="top_sentences" and embeddings=="pooling":
        return extracted_information,summarize_similarity_mean_pooling(extracted_information, number_of_sentences)
    if algorithm=="top_sentences" and embeddings=="sentenceTransformer":
        return extracted_information,summarize_similarity_sentenceTransformer(extracted_information, number_of_sentences)


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Information Extractor")

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Variables
        self.image_name = tk.StringVar()
        self.option = tk.StringVar(value="None")
        self.option2 = tk.StringVar(value="None")
        self.num_sentences = tk.IntVar(value=3)

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Radio buttons for algorithms
        self.cluster_radio = tk.Radiobutton(root, text="Cluster", variable=self.option, value="cluster")
        self.cluster_radio.pack(anchor=tk.W)

        self.top_sentences_radio = tk.Radiobutton(root, text="Top Sentences", variable=self.option, value="top_sentences")
        self.top_sentences_radio.pack(anchor=tk.W)

        # Radio buttons for embeddings
        self.cluster_radio = tk.Radiobutton(root, text="Pooling", variable=self.option2, value="pooling")
        self.cluster_radio.pack(anchor=tk.W)

        self.top_sentences_radio = tk.Radiobutton(root, text="Sentence Transformer", variable=self.option2, value="sentenceTransformer")
        self.top_sentences_radio.pack(anchor=tk.W)

        # Number of sentences input
        self.num_sentences_label = tk.Label(root, text="Number of sentences for summary:")
        self.num_sentences_label.pack(anchor=tk.W)

        self.num_sentences_entry = tk.Entry(root, textvariable=self.num_sentences)
        self.num_sentences_entry.pack(anchor=tk.W)

        # Extracted information display with scrollbar
        self.text_frame = tk.Frame(root)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        self.scrollbar = Scrollbar(self.text_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_display = tk.Text(self.text_frame, wrap=tk.WORD, yscrollcommand=self.scrollbar.set)
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.text_display.yview)

        # Button to load image
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Button to extract information
        self.extract_button = tk.Button(root, text="Extract Information", command=self.extract_info)
        self.extract_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_name.set(file_path)
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

    def extract_info(self):
        image_path = self.image_name.get()
        option = self.option.get()
        option2 = self.option2.get()
        num_sentences = self.num_sentences.get()
        if not image_path:
            messagebox.showerror("Error", "Please load an image first.")
            return
        if option == "None":
            messagebox.showerror("Error", "Please select an extraction option.")
            return
        if option2 == "None":
            messagebox.showerror("Error", "Please select an embedding option.")
            return
        extracted_text, summary = image_to_summary(image_path, num_sentences, option, option2)
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, f"Extracted Information:\n{extracted_text}\n\nSummary:\n{summary}")


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
