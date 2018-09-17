import re
from time import time

from nltk.stem import WordNetLemmatizer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")"
                              "|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


class Tokenizer(object):
    def __init__(self):
        self.lemma = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.lemma.lemmatize(i)
                for i in doc.split(' ')]


class ReviewSentimentAnalyzer(object):

    def __init__(self):
        self.clf_cls = LinearSVC
        self.y_true = [1 if i < 12500 else 0 for i in range(25000)]
        self.train()

    def preprocess_reviews(self, reviews):
        reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

        return reviews

    def get_reviews(self):
        print("Extracting text...")
        reviews_train = []
        for line in open('movie_data/full_train.txt', 'r'):
            reviews_train.append(line.strip())

        reviews_test = []
        for line in open('movie_data/full_test.txt', 'r'):
            reviews_test.append(line.strip())

        return (
            self.preprocess_reviews(reviews_train),
            self.preprocess_reviews(reviews_test)
        )

    def cross_validate(self, reviews):
        print("Finding best parameters...")
        self.tokenizer = Tokenizer()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            tokenizer=self.tokenizer,
            analyzer='word',
        )
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('clf', self.clf_cls())
        ])
        parameters = {
            'clf__C': (1,),
            'vectorizer__binary': (True,),
            'vectorizer__ngram_range': ((1, 2),),
            'vectorizer__lowercase': (True,),
        }
        self.grid_search = GridSearchCV(
            pipeline,
            param_grid=parameters,
            verbose=1,
            n_jobs=-1,
            refit=True
        )
        self.grid_search.fit(reviews, self.y_true)
        start = time()
        print("Hyperparameter search took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start),
                                        len(self.grid_search.grid_scores_)))
        print("Best score: %0.3f" % self.grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = self.grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        self.vectorizer = best_parameters['vectorizer']
        self.clf = best_parameters['clf']

    def train(self):
        print("Training...")
        reviews, reviews_test = self.get_reviews()
        self.cross_validate(reviews)
        y_pred = self.grid_search.predict(
            reviews_test
        )
        print("Final Accuracy: {}".format(
            accuracy_score(self.y_true, y_pred)))

    def sanity_check(self):
        print("Sanity checking...")
        feature_to_coef = {
            word: coef for word, coef in zip(
                self.vectorizer.get_feature_names(), self.clf.coef_[0]
            )
        }
        print("Top 5 positive words")
        for best_positive in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1],
                reverse=True)[:5]:
            print(best_positive)

        print("Top 5 negative words")
        for best_negative in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1])[:5]:
            print(best_negative)


def main():
    analyzer = ReviewSentimentAnalyzer()
    analyzer.sanity_check()


if __name__ == '__main__':
    main()
