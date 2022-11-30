# What is this Project about?
This Project was from Bangkit Capstone Project, the goal from this repository, is to built the sentiment analysis to classified the sentence is it the positive or the negative sentiment about the specific event organizer review.

# Data Used by this Model
In this project, we have `7049` data including the `label positive` and `label negative`. The data has been splited by two category: `Training data` and `Validation Data`. Training have `5639` Data and the Validation data have `1410` Data.

# Preprocessing
in the Processing phase, we using two methods: `Stopwords` and `Lowercasing`
```
def remove_stopwords(sentence):

    stopwords = ['a', 'the', ...]

    sentence = sentence.lower()

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
    return sentence
```

# Accuracy
<img src="https://github.com/Barbarpotato/Evenity_ML/blob/main/images/accuracy.png"></img>

# Loss
<img src="https://github.com/Barbarpotato/Evenity_ML/blob/main/images/loss.png"></img>

