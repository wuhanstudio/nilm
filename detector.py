class EdgeDetector:
    def __init__(self, current_time, current_measurement, state_threshold=15, noise_level=50, min_n_samples=2):
        # Hyperparameters
        self.state_threshold = state_threshold
        self.noise_level = noise_level
        self.min_n_samples = min_n_samples

        # Estimated Steady State Power
        self.N = 0
        self.estimated_steady_power = 0

        # Change detection flags
        self.ongoing_change = False          # power change in progress over multiple seconds
        self.instantaneous_change_queue = []
    
        # Transition time tracking
        self.tran_start_time = []
        self.tran_data_list = []
        self.tran_data = []
        self.tran_end_time = None

        self.index_transitions_start = []  
        self.index_transitions_end = []

        self.previous_time = current_time
        self.previous_measurement = current_measurement
        self.last_steady_power = current_measurement

        # Output data
        self.transitions = []                # holds information on transitions
        self.steady_states = []              # steadyStates to store in returned Dataframe
        self.index_steady_states = []

    def update(self, current_time, current_measurement):
        output = {'transition': False}

        # Step 2: this does the threshold test and then we sum the boolean
        state_change = abs(current_measurement - self.previous_measurement)

        instantaneous_change = False
        if  abs(current_measurement - self.last_steady_power) > self.noise_level:
            self.tran_start_time.append(self.previous_time)
            self.tran_data.append(self.previous_measurement)

        if state_change > self.state_threshold:
            instantaneous_change = True
        else:
            instantaneous_change = False

        # Identify end of ongoing change
        if self.ongoing_change:
            self.tran_data.append(self.previous_measurement)
            if not instantaneous_change:
                self.tran_end_time = self.previous_time

        # Step 3: Identify if transition is just starting, if so, process it
        # if instantaneous_change and (not ongoing_change):

        # Step 3: Identify if the state is now steady (response faster than Hart's algo)
        if len(self.instantaneous_change_queue) == self.min_n_samples and all(not x for x in self.instantaneous_change_queue):

            # Calculate transition size
            last_transition = self.estimated_steady_power - self.last_steady_power

            # Sum Boolean array to verify if transition is above noise level
            if abs(last_transition) > self.noise_level:
                # 3A, C: if so add the index of the transition start and the power information

                # Avoid outputting first transition from zero
                self.index_transitions_end.append(self.tran_end_time)
                self.index_transitions_start.append(self.tran_start_time[0])
                self.tran_start_time = []

                output['transition'] = True
                output['transition_start_time'] = self.index_transitions_start[-1]
                output['transition_end_time'] = self.index_transitions_end[-1]
                output['transition_power_change'] = last_transition
                output['transition_data'] = [ t for t in self.tran_data ]

                self.tran_data_list.append(self.tran_data)
                self.tran_data = []

                self.transitions.append(last_transition)

                # I think we want this, though not specifically in Hart's algo notes
                # We don't want to append a steady state if it's less than min samples in length.
                # if N > min_n_samples:
                self.index_steady_states.append(self.tran_end_time)

                # last states steady power
                self.steady_states.append(self.estimated_steady_power)
            # 3B
            self.last_steady_power = self.estimated_steady_power

        # 3C
        # tran_end_time = row.index[0]  # start new steady state

        # Step 4: if a new steady state is starting, zero counter
        if instantaneous_change:
            self.N = 0

        # Hart step 5: update our estimate for steady state's energy
        self.estimated_steady_power = (self.N * self.estimated_steady_power + current_measurement) / (
            self.N + 1
        )

        if len(self.instantaneous_change_queue) == self.min_n_samples:
            self.instantaneous_change_queue.pop(0)
        self.instantaneous_change_queue.append(instantaneous_change)

        # Step 6: increment counter
        self.N += 1

        # Step 7
        self.ongoing_change = instantaneous_change

        # Step 8
        self.previous_measurement = current_measurement
        self.previous_time = current_time

        return output