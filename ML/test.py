from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

train, train_labels = [], []
dataComb = []
i = 0

HEADER = ["review", "label"]

# pre processing dak usah peduli
# with open("train_data.csv", "r", encoding="utf8") as file:
#     data = csv.reader(file)
#     for list in data:
#         i += 1
#         if len(list) == 3 or len(list) == 1:
#             continue
#         dataTrain = list[1].encode("utf-8")
#         dataLabel = list[0].encode("utf-8")
#         dataTrain = dataTrain.decode("utf-8")
#         dataLabel = dataLabel.decode("utf-8")
#         train.append(dataTrain.strip())
#         train_labels.append(dataLabel.strip())
# print(i)

# for idx in range(0, len(train)):
#     dataComb.append([train[idx], train_labels[idx]])


# print(dataComb[0])

# with open("train_data2.csv", "w") as file:
#     writer = csv.writer(file)
#     writer.writerow(HEADER)
#     writer.writerows(dataComb)


def remove_stopwords(sentence):
    # List of stopwords
    stopwords = [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "nor",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "would",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]

    # Sentence converted to lowercase-only
    sentence = sentence.lower()

    ### START CODE HERE
    list_sentence, sentence = sentence.split(" "), ""
    set_sentence = set(list_sentence)

    for word in set_sentence:
        if word in stopwords:
            count_word = list_sentence.count(word)
            for count in range(0, count_word):
                list_sentence.remove(word)

    for word in list_sentence:
        sentence += word + " "

    sentence = sentence.rstrip()
    ### END CODE HERE
    return sentence


def parse_data_from_file(filename):
    dataFile = []
    sentences = []
    labels = []
    with open(filename, "r") as csvfile:
        ### START CODE HERE
        reader, skip_header = csv.reader(csvfile), True

        for shuffledata in reader:
            if skip_header == True:
                skip_header = False
                continue
            dataFile.append(shuffledata)

        random.shuffle(dataFile)

        for seperateData in dataFile:
            labels.append(seperateData[-1])
            filter_words = remove_stopwords(seperateData[0])
            sentences.append(filter_words)

        ### END CODE HERE
    return sentences, labels


def train_val_split(sentences, labels, training_split):

    ### START CODE HERE

    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = len(sentences) * training_split

    # Split the sentences and labels into train/validation splits
    train_sentences = sentences[: int(train_size)]
    train_labels = labels[: int(train_size)]

    validation_sentences = sentences[int(train_size) :]
    validation_labels = labels[int(train_size) :]

    ### END CODE HERE

    return train_sentences, validation_sentences, train_labels, validation_labels


def fit_tokenizer(train_sentences, num_words, oov_token):

    ### START CODE HERE

    # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    # Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)
    ### END CODE HERE

    return tokenizer


def seq_and_pad(sentences, tokenizer, padding, maxlen):

    ### START CODE HERE

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences using the correct padding and maxlen
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding)

    ### END CODE HERE

    return padded_sequences


def tokenize_labels(all_labels, split_labels):

    ### START CODE HERE

    # Instantiate the Tokenizer (no additional arguments needed)
    label_tokenizer = Tokenizer()

    # Fit the tokenizer on all the labels
    label_tokenizer.fit_on_texts(all_labels)

    # Convert labels to sequences
    label_seq = label_tokenizer.texts_to_sequences(split_labels)

    # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
    label_seq_np = np.array(label_seq) - 1

    ### END CODE HERE
    return label_seq_np


sentences, labels = parse_data_from_file("./data.csv")
# print(f"There are {len(sentences)} sentences in the dataset.\n")
# print(
#     f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n"
# )
# print(f"There are {len(labels)} labels in the dataset.\n")
# print(f"The first 5 labels are {set(labels)}")

# SPLIT THE DATA
train_sentences, val_sentences, train_labels, val_labels = train_val_split(
    sentences, labels, 0.8
)

print(len(train_sentences))
print(len(val_sentences))

# Tokenization training data
tokenizer = fit_tokenizer(train_sentences, 1000, "<OOV>")
word_index = tokenizer.word_index

# print(f"Vocabulary contains {len(word_index)} words\n")
# print(
#     "<OOV> token included in vocabulary"
#     if "<OOV>" in word_index
#     else "<OOV> token NOT included in vocabulary"
# )

# SEQUENCES
train_padded_sequences = seq_and_pad(train_sentences, tokenizer, "post", 120)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, "post", 120)
# print(f"First padded sequence looks like this: \n\n{padded_sequences[0]}\n")
# print(f"Numpy array of all sequences has shape: {padded_sequences.shape}\n")
# print(
#     f"This means there are {padded_sequences.shape[0]} sequences in total and each one has a size of {padded_sequences.shape[1]}"
# )

# Tokenization label
train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)

print(
    f"First 5 labels of the training set should look like this:\n{train_label_seq[:5]}\n"
)
print(
    f"First 5 labels of the validation set should look like this:\n{val_label_seq[:5]}\n"
)
print(f"Tokenized labels of the training set have shape: {train_label_seq.shape}\n")
print(f"Tokenized labels of the validation set have shape: {val_label_seq.shape}\n")


# CREATING THE MODEL
def create_model(num_words, embedding_dim, maxlen):

    tf.random.set_seed(123)

    ### START CODE HERE

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    ### END CODE HERE

    return model


model = create_model(1000, 16, 120)

history = model.fit(
    train_padded_sequences,
    train_label_seq,
    epochs=30,
    validation_data=(val_padded_seq, val_label_seq),
)


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f"val_{metric}"])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f"val_{metric}"])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


inputs = ["i love this event organizer"]

sequencePred = tokenizer.texts_to_sequences(
    inputs
)  # same tokenizer which is used on train data.
sequencePred = pad_sequences(sequencePred, maxlen=120)

predictions = model.predict(sequencePred)

print(predictions)


# convert to .tflite
tflite_model = tf.keras.models.load_model("model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
tflite_save = converter.convert()
open("generated.tflite", "wb").write(tflite_save)
