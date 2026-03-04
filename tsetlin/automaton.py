class Automaton:
    def __init__(self, N_state, state):
        assert N_state % 2 == 0, "N_state must be even"

        self.N_state = N_state
        self.middle_state = N_state // 2

        self.state = state
        self.action = self._action()

    def _action(self):
        return 1 if (self.state > (self.middle_state)) else 0

    def reward(self):
        self.state += 1
        self.action = self._action()

        # A new include literal
        return self.state == (self.middle_state + 1)

    def penalty(self):
        self.state -= 1
        self.action = self._action()

        # A new exclude literal
        return self.state == (self.middle_state)

    def update(self):
        self.action = self._action()
