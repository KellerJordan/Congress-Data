import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec


def test_func():
    return 'output A'

class Visualizer:
    
    def __init__(self, vec, cls):
        self.vec = vec
        self.cls = cls
        self.vec.inv_vocab = {k: w for w, k in self.vec.vocabulary_.items()}

    def process_sentence(self, row_text):
        # extract words from the sentence that are used by our model,
        # and compute the contribution  of each word to the final score.
        row_embed = self.vec.transform([row_text])
        indices = sp.find(row_embed)[1]
        words = [self.vec.inv_vocab[i] for i in indices]
        weights = np.array([row_embed[0, i] * self.cls.coef_[0][i] for i in indices])

        # test that the score from our weights matches that produced by the model.
        z = weights.sum() + self.cls.intercept_[0]
        pred = self.cls.predict_proba(row_embed)[0, 1]
        assert np.abs(1 / (1 + np.exp(-z)) - pred) < 1e-5

        return words, weights, pred

    def display_contributions(self, words, weights, pred):
        # split into positive and negative local sentiment
        word_pos, wgt_pos = zip(*[(word, w) for word, w in zip(words, weights) if w > 0])
        word_neg, wgt_neg = zip(*[(word, -w) for word, w in zip(words, weights) if w < 0])
        wgt_pos, wgt_neg = np.array(wgt_pos), np.array(wgt_neg)
        # convert to pie chart radius
        get_radius = lambda wgt: 1.5 * np.sqrt(wgt.sum()) / np.pi
        rad_pos, rad_neg = get_radius(wgt_pos), get_radius(wgt_neg)

        fig = plt.figure(figsize=(16, 8))
        # fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(15, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[8, 1, 8])

        ax1 = plt.subplot(gs[0])
        ax1.pie(wgt_pos / wgt_pos.sum(), labels=word_pos, radius=rad_pos)
        ax1.set_title('Positive contributions to log-odds')

        ax2 = plt.subplot(gs[2])
        ax2.pie(wgt_neg / wgt_neg.sum(), labels=word_neg, radius=rad_neg)
        ax2.set_title('Negative contributions to log-odds')

        ev = np.log(pred / (1 - pred))
        ax3 = plt.subplot(gs[1])
        ax3.bar([0], [ev], width=0.01, color=['red', 'green'][int(ev > 0)])
        ax3.set_ylim([-5, 5])
        ax3.set_title('Predicted log(p/1-p)')
        ax3.set_xticklabels([])
        ax3.set_xticks([])

        plt.show()

    def viz_sentence(self, row_text):
        words, weights, pred = self.process_sentence(row_text)
        self.display_contributions(words, weights, pred)
        
    def word_bias(self, word, dset, tgt):
        numer, denom = 0, 0
        for i, s in enumerate(dset):
            if word in s:
                denom += 1
                if tgt[i] == 1.0:
                    numer += 1
        prob = numer / denom
        return prob
