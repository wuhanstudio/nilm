
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

    def compress(self, compressed=False):
        self.p_included_mask = bitarray(self.N_feature)
        self.n_included_mask = bitarray(self.N_feature)

        if compressed:
            # Get the index of trainable literals
            for i in self.p_trainable_literals:
                if self.p_automata[i].action == 1:
                    self.p_included_mask[i] = 1
            for i in self.n_trainable_literals:
                if self.n_automata[i].action == 1:
                    self.n_included_mask[i] = 1
        else:
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

    def set_state(self, states):
        assert len(states) == 2 * self.N_feature, "States must be a 1D array with shape (2 * N_features)"

        for i in range(self.N_feature):
            self.p_automata[i].state = states[i]
            self.p_automata[i].update()
            self.n_automata[i].state = states[i + self.N_feature]
            self.n_automata[i].update()

        self.compress()

    def get_state(self):
        p_states = [a.state for a in self.p_automata]
        n_states = [a.state for a in self.n_automata]

        return p_states + n_states

    def set_compressed_state(self, n_pos_leteral, n_neg_literal, positions, states):
            assert len(states) == len(positions), "Length of states must be equal to length of positions"
            assert n_pos_leteral + n_neg_literal == len(positions), "Number of trainable literals does not match length of positions"

            self.p_automata = {}
            self.n_automata = {}

            for i in range(n_pos_leteral):
                pos = positions[i]

                self.p_automata[pos] = Automaton(self.N_states, -1)
                self.p_automata[pos].state = states[i]
                self.p_automata[pos].update()

            for i in range(n_neg_literal):
                pos = positions[i + n_pos_leteral]

                self.n_automata[pos] = Automaton(self.N_states, -1)
                self.n_automata[pos].state = states[i + n_pos_leteral]
                self.n_automata[pos].update()

            self.compress(compressed=True)

    def get_compressed_state(self):
        p_states = [self.p_automata[a].state for a in self.p_trainable_literals]
        n_states = [self.n_automata[a].state for a in self.n_trainable_literals]

        return p_states + n_states
