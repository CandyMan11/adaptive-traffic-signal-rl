import traci
import numpy as np


class TrafficEnv:
    def __init__(self):

        self.sumo_cmd = [
            "sumo",
            "-c",
            "sumo_env/config/simulation.sumocfg",
            "--start"
        ]

        self.tls_id = None
        self.prev_waiting = 0

        # Minimum green time
        self.min_green = 5
        self.phase_timer = 0
        self.last_action = None

        self.max_steps = 200
        self.current_step = 0

        self.num_phases = 4  # we define 4 custom phases

    # --------------------------------------------------
    # Start SUMO
    # --------------------------------------------------
    def start(self):
        traci.start(self.sumo_cmd)

        tls_list = traci.trafficlight.getIDList()
        self.tls_id = tls_list[0]

        self._install_tls_logic()

        self.prev_waiting = 0
        self.current_step = 0
        self.phase_timer = 0
        self.last_action = None

    # --------------------------------------------------
    # Install custom 4-phase traffic light logic
    # --------------------------------------------------
    def _install_tls_logic(self):

        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]

        phases = []
        phases.append(traci.trafficlight.Phase(30, "GrGr"))
        phases.append(traci.trafficlight.Phase(5, "yryr"))
        phases.append(traci.trafficlight.Phase(30, "rGrG"))
        phases.append(traci.trafficlight.Phase(5, "ryry"))

        new_logic = traci.trafficlight.Logic(
            logic.programID,
            logic.type,
            logic.currentPhaseIndex,
            phases
        )

        traci.trafficlight.setProgramLogic(self.tls_id, new_logic)

    # --------------------------------------------------
    # State Representation (Normalized + Phase info)
    # --------------------------------------------------
    def get_state(self):

        lanes = traci.lane.getIDList()

        waiting_times = []

        for lane in lanes:
            waiting = traci.lane.getWaitingTime(lane)
            # Normalize waiting (assume max reasonable ~100 sec)
            waiting_times.append(waiting / 100.0)

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # Normalize phase index
        phase_normalized = current_phase / (self.num_phases - 1)

        state = waiting_times + [phase_normalized]

        return np.array(state, dtype=np.float32)

    # --------------------------------------------------
    # Step Function
    # --------------------------------------------------
    def step(self, action):

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        switch_penalty = 0

        # Enforce minimum green
        if action != current_phase and self.phase_timer >= self.min_green:
            traci.trafficlight.setPhase(self.tls_id, action)
            self.phase_timer = 0

            # Penalize switching
            if self.last_action is not None and action != self.last_action:
                switch_penalty = -0.2

            self.last_action = action
        else:
            self.phase_timer += 1

        # Advance simulation
        traci.simulationStep()
        self.current_step += 1

        # Compute total waiting
        total_waiting = 0
        vehicle_ids = traci.vehicle.getIDList()

        for vid in vehicle_ids:
            total_waiting += traci.vehicle.getWaitingTime(vid)

        # Delta waiting reward (normalized)
        delta_waiting = (self.prev_waiting - total_waiting) / 10.0
        reward = delta_waiting + switch_penalty

        self.prev_waiting = total_waiting

        state = self.get_state()

        done = self.current_step >= self.max_steps

        return state, reward, done

    # --------------------------------------------------
    # Reset Environment
    # --------------------------------------------------
    def reset(self):
        traci.close()
        self.start()
        return self.get_state()

    # --------------------------------------------------
    # Close SUMO
    # --------------------------------------------------
    def close(self):
        traci.close()