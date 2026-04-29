import random

import numpy as np
from tqdm import tqdm

from tsetlin.clause import Clause

def to_int32(val):
    val = int(val)
    return max(-2**31, min(val, 2**31 - 1))

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

    @staticmethod
    def load_model(path):
        import tsetlin_pb2
        tm = tsetlin_pb2.Tsetlin()

        with open(path, "rb") as f:
            tm.ParseFromString(f.read())

        tm_model = Tsetlin(N_feature=tm.n_feature, N_class=tm.n_class, N_clause=tm.n_clause, N_state=tm.n_state)
        tm_model.n_classes = tm.n_class
        tm_model.n_features = tm.n_feature
        tm_model.n_clauses = tm.n_clause
        tm_model.n_states = tm.n_state

        if tm.model_type == tsetlin_pb2.ModelType.INFERENCE:
            tm_model.pos_clauses = []
            tm_model.neg_clauses = []
            for i in range(tm_model.n_classes):
                pos_clauses = []
                neg_clauses = []
                for j in range(tm_model.n_clauses // 2):
                    p_clause_c = tm.clauses_compressed[i * tm_model.n_clauses + j * 2]
                    n_clause_c = tm.clauses_compressed[i * tm_model.n_clauses + j * 2 + 1]

                    # Set positive clauses
                    pos_clause = Clause(tm_model.n_features, tm_model.n_states)

                    pos_clause.p_trainable_literals = p_clause_c.position[:p_clause_c.n_pos_literal]
                    pos_clause.n_trainable_literals = p_clause_c.position[p_clause_c.n_pos_literal:]
                    
                    # Defaults to middle state + 1
                    pos_clause.set_compressed_state(p_clause_c.n_pos_literal, p_clause_c.n_neg_literal, p_clause_c.position, [tm_model.n_states // 2 + 1] * (p_clause_c.n_pos_literal + p_clause_c.n_neg_literal))
                    pos_clauses.append(pos_clause)

                    # Set negative clauses
                    neg_clause = Clause(tm_model.n_features, tm_model.n_states)

                    neg_clause.p_trainable_literals = n_clause_c.position[:n_clause_c.n_pos_literal]
                    neg_clause.n_trainable_literals = n_clause_c.position[n_clause_c.n_pos_literal:]

                    # Defaults to middle state + 1
                    neg_clause.set_compressed_state(n_clause_c.n_pos_literal, n_clause_c.n_neg_literal, n_clause_c.position, [tm_model.n_states // 2 + 1] * (n_clause_c.n_pos_literal + n_clause_c.n_neg_literal))
                    neg_clauses.append(neg_clause)

                tm_model.pos_clauses.append(pos_clauses)
                tm_model.neg_clauses.append(neg_clauses)

        elif tm.model_type == tsetlin_pb2.ModelType.TRAINING:
            tm_model.pos_clauses = []
            tm_model.neg_clauses = []
            for i in range(tm_model.n_classes):
                pos_clauses = []
                neg_clauses = []
                for j in range(tm_model.n_clauses // 2):
                    p_clause = tm.clauses[i * tm_model.n_clauses + j * 2]
                    n_clause = tm.clauses[i * tm_model.n_clauses + j * 2 + 1]

                    # Set positive clauses
                    pos_clause = Clause(tm_model.n_features, tm_model.n_states)
                    pos_clause.set_state(p_clause.data)
                    pos_clauses.append(pos_clause)

                    # Set negative clauses
                    neg_clause = Clause(tm_model.n_features, tm_model.n_states)
                    neg_clause.set_state(n_clause.data)
                    neg_clauses.append(neg_clause)

                tm_model.pos_clauses.append(pos_clauses)
                tm_model.neg_clauses.append(neg_clauses)

        return tm_model

    def save_model(self, path, type="training"):
        import tsetlin_pb2
        tm = tsetlin_pb2.Tsetlin()

        tm.n_class = self.n_classes
        tm.n_feature = self.n_features
        tm.n_clause = self.n_clauses
        tm.n_state = self.n_states

        if type not in ["training", "inference"]:
            raise ValueError("type must be either 'training' or 'inference'")

        if type == "training":
            tm.model_type = tsetlin_pb2.ModelType.TRAINING
        elif type == "inference":
            tm.model_type = tsetlin_pb2.ModelType.INFERENCE

        for i in range(self.n_classes):
            for j in range(self.n_clauses // 2):
                if type == "inference":
                    # Positive clauses
                    p_inference_literals = []
                    n_inference_literals = []
                    for k in range(self.n_features):
                        if self.pos_clauses[i][j].p_automata[k].state > (self.n_states // 2):
                            p_inference_literals.append(k)
                        if self.pos_clauses[i][j].n_automata[k].state > (self.n_states // 2):
                            n_inference_literals.append(k)

                    pos_c = tsetlin_pb2.ClauseCompressed()
                    pos_c.n_pos_literal = len(p_inference_literals)
                    pos_c.n_neg_literal = len(n_inference_literals)

                    pos_positions = [to_int32(x) for x in p_inference_literals]
                    pos_c.position.extend(pos_positions)

                    neg_positions = [to_int32(x) for x in n_inference_literals]
                    pos_c.position.extend(neg_positions)

                    # States data not requried for inference
                    # pos_c.data.extend([to_int32(x) for x in self.pos_clauses[i][j].get_compressed_state()])

                    tm.clauses_compressed.append(pos_c)

                    # Negative clauses
                    p_inference_literals = []
                    n_inference_literals = []
                    for k in range(self.n_features):
                        if self.neg_clauses[i][j].p_automata[k].state > (self.n_states // 2):
                            p_inference_literals.append(k)
                        if self.neg_clauses[i][j].n_automata[k].state > (self.n_states // 2):
                            n_inference_literals.append(k)

                    neg_c = tsetlin_pb2.ClauseCompressed()
                    neg_c.n_pos_literal = len(p_inference_literals)
                    neg_c.n_neg_literal = len(n_inference_literals)

                    pos_positions = [to_int32(x) for x in p_inference_literals]
                    neg_c.position.extend(pos_positions)

                    neg_positions = [to_int32(x) for x in n_inference_literals]
                    neg_c.position.extend(neg_positions)

                    # States data not requried for inference
                    # neg_c.data.extend([to_int32(x) for x in self.neg_clauses[i][j].get_compressed_state()])

                    tm.clauses_compressed.append(neg_c)

                elif type == "training":
                    # Positive clauses
                    pos_c = tsetlin_pb2.Clause()
                    pos_c.data.extend([to_int32(x) for x in self.pos_clauses[i][j].get_state()])

                    tm.clauses.append(pos_c)

                    # Negative clauses
                    neg_c = tsetlin_pb2.Clause()
                    neg_c.data.extend([to_int32(x) for x in self.neg_clauses[i][j].get_state()])

                    tm.clauses.append(neg_c)

        with open(path, "wb") as f:
            f.write(tm.SerializeToString())
