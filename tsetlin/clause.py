
import random

import fastrand
rand = fastrand.pcg32_uniform

from bitarray import bitarray

from tsetlin.automaton import Automaton

class Clause:
    def __init__(self, N_feature, N_state):

        assert N_state % 2 == 0, "N_state must be even"

        self.N_feature = N_feature
        self.N_states = N_state
        self.N_literals = 2 * N_feature

        # Positive and Negative Automata for each feature
        self.p_automata = [Automaton(N_state, -1) for _ in range(N_feature)]
        self.n_automata = [Automaton(N_state, -1) for _ in range(N_feature)]

        # Randomly initialize automata states middle_state + {0,1}
        for i in range(self.N_feature):
            choice = random.choice([0, 1])
            self.p_automata[i].state = N_state // 2 + choice
            self.n_automata[i].state = N_state // 2 + (1 - choice)

        self.compress()

    def compress(self):
        self.p_included_mask = bitarray(self.N_feature)
        self.n_included_mask = bitarray(self.N_feature)

        # Get the index of included literals
        for i in range(self.N_feature):
            if self.p_automata[i].action == 1:
                self.p_included_mask[i] = 1
            if self.n_automata[i].action == 1:
                self.n_included_mask[i] = 1

    def evaluate(self, X):
        # Evaluate with compression (faster)
        if not isinstance(X, bitarray):
            X = bitarray(list(map(bool, X)))
        if X & self.p_included_mask != self.p_included_mask:
            return 0
        if (~X) & self.n_included_mask != self.n_included_mask:
            return 0

        return 1

    def type_I_feedback(self, X, clause_output, s):
        feedback_count = 0

        # Want clause_output to be 1
        s1 = 1 / s
        s2 = (s - 1) / s

        # Erase Pattern
        # Reduce the number of included literals
        if clause_output == 0:
            # Positive literal X
            for i in range(self.N_feature) :
                if self.p_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    if self.p_automata[i].penalty():
                        self.p_included_mask[i] = 0

            # Negative literal NOT X
            for i in range(self.N_feature):
                if self.n_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    if self.n_automata[i].penalty():
                        self.n_included_mask[i] = 0

        # Recognize Pattern
        # Increase the number of included literals
        if clause_output == 1:
            # Positive literal X
            for i in range(self.N_feature):
                if X[i] == 1 and self.p_automata[i].state < self.N_states and rand() <= s2:
                    feedback_count += 1
                    if self.p_automata[i].reward():
                        self.p_included_mask[i] = 1

                elif X[i] == 0 and self.p_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    if self.p_automata[i].penalty():
                        self.p_included_mask[i] = 0

            # Negative literal NOT X
            for i in range(self.N_feature):
                if X[i] == 1 and self.n_automata[i].state > 1 and rand() <= s1:
                    feedback_count += 1
                    if self.n_automata[i].penalty():
                        self.n_included_mask[i] = 0

                elif X[i] == 0 and self.n_automata[i].state < self.N_states and rand() <= s2:
                    feedback_count += 1
                    if self.n_automata[i].reward():
                        self.n_included_mask[i] = 1

        return feedback_count

    def type_II_feedback(self, X):
        feedback_count = 0
        # Want clause_output to be 0
        for i in range(self.N_feature):
            if (X[i] == 0) and (self.p_included_mask[i] == 0): 
                feedback_count += 1
                if self.p_automata[i].reward():
                    self.p_included_mask[i] = 1

        for i in range(self.N_feature):
            if (X[i] == 1) and (self.n_included_mask[i] == 0):
                feedback_count += 1
                if self.n_automata[i].reward():
                    self.n_included_mask[i] = 1

        return feedback_count
