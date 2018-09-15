import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")"
                              "|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
TARGET = [1 if i < 12500 else 0 for i in range(25000)]


class ReviewSentimentAnalyzer(object):

    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)
        self.train()

    def preprocess_reviews(self, reviews):
        reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

        return reviews

    def vectorize(self, train, test):
        self.vectorizer.fit(train)
        X = self.vectorizer.transform(train)
        X_test = self.vectorizer.transform(test)
        return X, X_test

    def get_train_test_data(self):
        reviews_train = []
        for line in open('movie_data/full_train.txt', 'r'):
            reviews_train.append(line.strip())

        reviews_test = []
        for line in open('movie_data/full_test.txt', 'r'):
            reviews_test.append(line.strip())

        return self.vectorize(
            self.preprocess_reviews(reviews_train),
            self.preprocess_reviews(reviews_test)
        )

    def get_hyper_parameters(self, X):
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

    def train(self):
        X, X_test = self.get_train_test_data()
        c = self.get_hyper_parameters(X)
        self.clf = LogisticRegression(C=c)
        self.clf.fit(X, TARGET)
        print("Final Accuracy: {}".format(
            accuracy_score(TARGET, self.clf.predict(X_test))))

    def sanity_check(self):
        feature_to_coef = {
            word: coef for word, coef in zip(
                self.vectorizer.get_feature_names(), self.clf.coef_[0]
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
    analyzer = ReviewSentimentAnalyzer()
    analyzer.sanity_check()


if __name__ == '__main__':
    main()
