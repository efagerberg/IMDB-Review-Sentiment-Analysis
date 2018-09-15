import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
TARGET = [1 if i < 12500 else 0 for i in range(25000)]


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews


def vectorize(train, test):
    cv = CountVectorizer(binary=True)
    cv.fit(train)
    X = cv.transform(train)
    X_test = cv.transform(test)
    return X, X_test, cv


def get_train_test_data():
    reviews_train = []
    for line in open('movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())

    reviews_test = []
    for line in open('movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())

    return vectorize(
        preprocess_reviews(reviews_train),
        preprocess_reviews(reviews_test)
    )


def get_hyper_parameters(X):
    X_train, X_val, y_train, y_val = train_test_split(
        X, TARGET, train_size=0.75
    )

    accuracy_scores = {}

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        a_score = accuracy_score(y_val, lr.predict(X_val))
        print("Accuracy for C={}: {}".format(c, a_score))
        accuracy_scores[c] = a_score

    return max(accuracy_scores, key=accuracy_scores.get)


def train_final_model(X, X_test, c):
    final_model = LogisticRegression(C=c)
    final_model.fit(X, TARGET)
    print("Final Accuracy: {}".format(
        accuracy_score(TARGET, final_model.predict(X_test))))
    return final_model


def sanity_check(final_model, cv):
    feature_to_coef = {
        word: coef for word, coef in zip(
            cv.get_feature_names(), final_model.coef_[0]
        )
    }
    for best_positive in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1],
            reverse=True)[:5]:
        print(best_positive)

    for best_negative in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1])[:5]:
        print(best_negative)


def main():
    X, X_test, cv = get_train_test_data()
    c = get_hyper_parameters(X)
    final_model = train_final_model(X, X_test, c)
    sanity_check(final_model, cv)


if __name__ == '__main__':
    main()
