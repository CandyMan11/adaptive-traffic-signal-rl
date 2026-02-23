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

        # Minimum green time control
        self.min_green = 5
        self.phase_timer = 0

        self.max_steps = 200
        self.current_step = 0

    # -----------------------------
    # Start SUMO
    # -----------------------------
    def start(self):
        traci.start(self.sumo_cmd)

        # Get traffic light ID
        tls_list = traci.trafficlight.getIDList()
        self.tls_id = tls_list[0]

        # Install custom 4-phase logic
        self._install_tls_logic()

        self.prev_waiting = 0
        self.current_step = 0
        self.phase_timer = 0

    # -----------------------------
    # Install 4-phase traffic light
    # -----------------------------
    def _install_tls_logic(self):

        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = []

        # 4 phases (adjust if needed)
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

    # -----------------------------
    # Get state representation
    # -----------------------------
    def _get_state(self):

        vehicle_ids = traci.vehicle.getIDList()

        total_waiting = 0
        vehicle_count = len(vehicle_ids)

        for vid in vehicle_ids:
            total_waiting += traci.vehicle.getWaitingTime(vid)

        avg_wait = total_waiting / vehicle_count if vehicle_count > 0 else 0

        return np.array([vehicle_count, avg_wait], dtype=np.float32)

    # -----------------------------
    # Environment Step
    # -----------------------------
    def step(self, action):

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # Apply minimum green constraint
        if action != current_phase and self.phase_timer >= self.min_green:
            traci.trafficlight.setPhase(self.tls_id, action)
            self.phase_timer = 0
        else:
            self.phase_timer += 1

        # Advance simulation
        traci.simulationStep()
        self.current_step += 1

        # Calculate total waiting
        total_waiting = 0
        vehicle_ids = traci.vehicle.getIDList()

        for vid in vehicle_ids:
            total_waiting += traci.vehicle.getWaitingTime(vid)

        # Delta waiting reward (normalized)
        reward = (self.prev_waiting - total_waiting) / 10.0
        self.prev_waiting = total_waiting

        state = self._get_state()

        done = self.current_step >= self.max_steps

        return state, reward, done

    # -----------------------------
    # Reset Environment
    # -----------------------------
    def reset(self):
        traci.close()
        self.start()
        return self._get_state()

    # -----------------------------
    # Close SUMO
    # -----------------------------
    def close(self):
        traci.close()