import numpy as np
import lda
import lda.datasets
import time



def main():
    X = lda.datasets.load_reuters()
    vocab = lda.datasets.load_reuters_vocab()
    titles = lda.datasets.load_reuters_titles()
    fit_baseline(X)



def fit_baseline(X, n_topics=20, n_iter=1500):
    """Fits an LDA model on feature matrix X and prints how long it took"""
    model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
    with Timer() as t:
        model.fit(X)
    print 'Fit %d articles with a vocab of %d words in %.3f seconds' % (X.shape + (t.interval, ))
    return model

#  time class from http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
#  and lots of other places on the web
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

if __name__ == '__main__':
    main()