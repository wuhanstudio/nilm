import random

import numpy as np
from tqdm import tqdm

from tsetlin.clause import Clause

class Tsetlin:
    def __init__(self, N_feature, N_class, N_clause, N_state):

        assert N_state % 2 == 0, "N_state must be even"
        assert N_clause % 2 == 0, "N_clause must be even"

        self.n_features = N_feature
        self.n_classes = N_class

        self.n_clauses = N_clause
        self.n_states = N_state

        self.pos_clauses = []
        self.neg_clauses = []
        for _ in range(N_class):
            self.pos_clauses.append([Clause(N_feature, N_state=N_state) for _ in range(int(N_clause / 2))])
            self.neg_clauses.append([Clause(N_feature, N_state=N_state) for _ in range(int(N_clause / 2))])

    def predict(self, X, return_votes=False):
        y_pred = [0] * len(X)
        n_half = int(self.n_clauses / 2)

        votes_list = []
        for i in tqdm(range(len(X)), desc="Predicting", ascii=True):
            Xi = X[i]
            votes = [0] * self.n_classes
            for c in range(self.n_classes):
                pos_clause = self.pos_clauses[c]
                neg_clause = self.neg_clauses[c]
                for j in range(n_half):
                    votes[c] += pos_clause[j].evaluate(Xi)
                    votes[c] -= neg_clause[j].evaluate(Xi)

            y_pred[i] = np.argmax(votes)

            if return_votes:
                votes_list.append(votes)

        if return_votes:
            return y_pred, votes_list
        else:
            return y_pred

    def step(self, X, y_target, T, s):
        # Pair-wise learning

        # Pair 1: Target class
        class_sum = 0
        pos_clauses = [0] * int(self.n_clauses / 2)
        neg_clauses = [0] * int(self.n_clauses / 2)
        for i in range(int(self.n_clauses / 2)):
            pos_clauses[i] = self.pos_clauses[y_target][i].evaluate(X)
            neg_clauses[i] = self.neg_clauses[y_target][i].evaluate(X)
            class_sum += pos_clauses[i]
            class_sum -= neg_clauses[i]

        # Clamp class_sum to [-T, T]
        class_sum = np.clip(class_sum, -T, T)
    
        # Calculate probabilities
        c1 = (T - class_sum) / (2 * T)

        # Update clauses for the target class
        for i in range(int(self.n_clauses / 2)):
            # Positive Clause: Type I Feedback
            if (random.random() <= c1):
                feedback_count = self.pos_clauses[y_target][i].type_I_feedback(X, pos_clauses[i], s=s)

            # Negative Clause: Type II Feedback
            if neg_clauses[i] == 1 and (random.random() <= c1):
                feedback_count = self.neg_clauses[y_target][i].type_II_feedback(X)

        # Pair 2: Non-target classes
        other_class = random.choice([x for x in range(self.n_classes) if x != y_target])

        class_sum = 0
        pos_clauses = [0] * int(self.n_clauses / 2)
        neg_clauses = [0] * int(self.n_clauses / 2)
        for i in range(int(self.n_clauses / 2)):
            pos_clauses[i] = self.pos_clauses[other_class][i].evaluate(X)
            neg_clauses[i] = self.neg_clauses[other_class][i].evaluate(X)
            class_sum += pos_clauses[i]
            class_sum -= neg_clauses[i]

        # Clamp class_sum to [-T, T]
        class_sum = np.clip(class_sum, -T, T)

        # Calculate probabilities
        c2 = (T + class_sum) / (2 * T)
        for i in range(int(self.n_clauses / 2)):
            # Positive Clause: Type II Feedback
            if pos_clauses[i] == 1 and (random.random() <= c2):
                feedback_count = self.pos_clauses[other_class][i].type_II_feedback(X)

            # Negative Clause: Type I Feedback
            if (random.random() <= c2):
                feedback_count = self.neg_clauses[other_class][i].type_I_feedback(X, neg_clauses[i], s=s)
