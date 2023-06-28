def word2features(sent, i, language):
    word = sent[i][0]
    features = {
        'word': word,
        'word.isdigit()': word.isdigit()
    }
    if language == 'English':
        features.update({
            'word.lower()': word.lower(),
            'word[:3]': word[:3],
            'word[:2]': word[:2],
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
        })
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1_word': word1,
            '-1_0_word': word1 + '_' + word,
            '-1:word.isdigit()': word1.isdigit()
        })
        if language == 'English':
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1word[:3]': word1[:3],
                '-1word[:2]': word1[:2],
                '-1word[-3:]': word1[-3:],
                '-1word[-2:]': word1[-2:],
                '-1word.isupper()': word1.isupper(),
                '-1:word.istitle()': word1.istitle()
            })
    else:
        features['BOS'] = True
    if i > 1:
        word1 = sent[i - 2][0]
        features.update({
            '-2_word': word1,
            '-2_-1_word': word1 + '_' + sent[i - 1][0],
            '-2:word.isdigit()': word1.isdigit()
        })
        if language == 'English':
            features.update({
                '-2:word.lower()': word1.lower(),
                '-2word[:3]': word1[:3],
                '-2word[:2]': word1[:2],
                '-2word[-3:]': word1[-3:],
                '-2word[-2:]': word1[-2:],
                '-2word.isupper()': word1.isupper(),
                '-2:word.istitle()': word1.istitle(),
            })
    else:
        features['BOS2'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1_word': word1,
            '0_+1_word': word + '_' + word1,
            '+1:word.isdigit()': word1.isdigit()
        })
        if language == 'English':
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1word[:3]': word1[:3],
                '+1word[:2]': word1[:2],
                '+1word[-3:]': word1[-3:],
                '+1word[-2:]': word1[-2:],
                '+1word.isupper()': word1.isupper(),
                '+1:word.istitle()': word1.istitle(),
            })
    else:
        features['EOS'] = True
    if i < len(sent) - 2:
        word1 = sent[i + 2][0]
        features.update({
            '+2_word': word1,
            '+1_+2+word': sent[i + 1][0] + '_' + word1,
            '+2:word.isdigit()': word1.isdigit()
        })
        if language == 'English':
            features.update({
                '+2:word.lower()': word1.lower(),
                '+2word[:3]': word1[:3],
                '+2word[:2]': word1[:2],
                '+2word[-3:]': word1[-3:],
                '+2word[-2:]': word1[-2:],
                '+2word.isupper()': word1.isupper(),
                '+2:word.istitle()': word1.istitle(),
            })
    else:
        features['EOS2'] = True
    if 0 < i < len(sent) - 1:
        features.update({
            '-1_+1_word': sent[i - 1][0] + '_' + sent[i + 1][0],
        })
    else:
        features['NOT_MIDDLE'] = True
    return features


def extract_features(sentences, language):
    X = []
    for sent in sentences:
        x = [word2features(sent, i, language) for i in range(len(sent))]
        X.append(x)
    return X


def extract_labels(sentences):
    y = []
    for sent in sentences:
        labels = [label for word, label in sent]
        y.append(labels)
    return y
