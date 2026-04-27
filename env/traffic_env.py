import traci
import numpy as np


class TrafficEnv:
    def __init__(self, gui=False):

        sumo_binary = "sumo-gui" if gui else "sumo"

        self.sumo_cmd = [
            sumo_binary,
            "-c",
            "sumo_env/config/simulation.sumocfg",
            "--start"
        ]

        self.tls_id = None
        self.prev_waiting = 0

        # Traffic light control
        self.min_green = 5
        self.phase_timer = 0
        self.num_phases = 4

        # Simulation control
        self.max_steps = 200
        self.current_step = 0

        # Action tracking
        self.last_action = None

    # -----------------------------
    # Start SUMO
    # -----------------------------
    def start(self):
        traci.start(self.sumo_cmd)

        self.tls_id = traci.trafficlight.getIDList()[0]
        self._install_tls_logic()

        self.prev_waiting = 0
        self.current_step = 0
        self.phase_timer = 0
        self.last_action = None

    # -----------------------------
    # Traffic Light Logic
    # -----------------------------
    def _install_tls_logic(self):

        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]

        phases = [
            traci.trafficlight.Phase(30, "GrGr"),
            traci.trafficlight.Phase(5, "yryr"),
            traci.trafficlight.Phase(30, "rGrG"),
            traci.trafficlight.Phase(5, "ryry"),
        ]

        new_logic = traci.trafficlight.Logic(
            logic.programID,
            logic.type,
            logic.currentPhaseIndex,
            phases
        )

        traci.trafficlight.setProgramLogic(self.tls_id, new_logic)

    # -----------------------------
    # STATE (Normalized)
    # -----------------------------
    def get_state(self):

        lanes = traci.lane.getIDList()
        lane_waits = []

        for lane in lanes:
            waiting = traci.lane.getWaitingTime(lane)
            lane_waits.append(waiting / 50.0)   # normalize

        phase = traci.trafficlight.getPhase(self.tls_id)

        return np.array(
            lane_waits + [phase / self.num_phases],
            dtype=np.float32
        )

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # Minimum green constraint
        if action != current_phase and self.phase_timer >= self.min_green:
            traci.trafficlight.setPhase(self.tls_id, action)
            self.phase_timer = 0
        else:
            self.phase_timer += 1

        # Advance simulation
        traci.simulationStep()
        self.current_step += 1

        vehicle_ids = traci.vehicle.getIDList()

        # -------------------------
        # 🚑 Emergency Override
        # -------------------------
        for vid in vehicle_ids:
            if "ambulance" in vid:
                print("🚑 Emergency override activated")
                traci.trafficlight.setPhase(self.tls_id, 0)

        # -------------------------
        # Waiting Calculation
        # -------------------------
        total_waiting = 0
        lane_waits = []

        for lane in traci.lane.getIDList():
            w = traci.lane.getWaitingTime(lane)
            lane_waits.append(w)
            total_waiting += w

        # -------------------------
        # ⚖️ FAIRNESS PENALTY
        # -------------------------
        max_wait = max(lane_waits) if lane_waits else 0
        avg_wait = np.mean(lane_waits) if lane_waits else 0

        fairness_penalty = (max_wait - avg_wait) * 0.2

        # -------------------------
        # 🔁 Reward (Delta + Fairness)
        # -------------------------
        # -------------------------
        # ⏰ RUSH HOUR DETECTION
        # -------------------------
        if 70 <= self.current_step <= 140:
            rush_multiplier = 1.5   # heavy penalty during peak
            print("🔥 Rush Hour Active")
        else:
            rush_multiplier = 1.0

        # -------------------------
        # 🔁 Reward (Adaptive)
        # -------------------------
        delta = (self.prev_waiting - total_waiting) / 10.0

        reward = (delta * rush_multiplier) - (fairness_penalty * rush_multiplier)

        self.prev_waiting = total_waiting

        # -------------------------
        # Done
        # -------------------------
        done = self.current_step >= self.max_steps

        return self.get_state(), reward, done

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        try:
            traci.close()
        except:
            pass

        self.start()
        return self.get_state()

    # -----------------------------
    # CLOSE
    # -----------------------------
    def close(self):
        try:
            traci.close()
        except:
            pass