import os
import sys
import traci

print("Main file executed")

# Check SUMO_HOME
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME environment variable")

# Path to config file
sumo_config = "sumo_env/config/simulation.sumocfg"

# Use SUMO without GUI for control
sumo_binary = "sumo"

# Start SUMO in TraCI mode
traci.start([
    sumo_binary,
    "-c", sumo_config,
    "--step-length", "1"
])

print("SUMO started successfully!")

step = 0
max_steps = 200

while step < max_steps:
    traci.simulationStep()
    step += 1

    print(f"\nPython Step: {step}")

    tls_ids = traci.trafficlight.getIDList()

    if tls_ids:
        tls_id = tls_ids[0]

        # Get current phase
        current_phase = traci.trafficlight.getPhase(tls_id)

        # Get number of phases (compatible with SUMO 1.26)
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phase_count = len(logic.phases)

        # Switch phase every 50 steps
        if step % 50 == 0:
            new_phase = (current_phase + 1) % phase_count
            traci.trafficlight.setPhase(tls_id, new_phase)
            print(f"Switched to phase {new_phase}")

        lanes = traci.trafficlight.getControlledLanes(tls_id)

        total_waiting_time = 0
        total_vehicles = 0

        for lane in lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_vehicles += traci.lane.getLastStepVehicleNumber(lane)

        print(
            f"Phase: {current_phase} | "
            f"Vehicles: {total_vehicles} | "
            f"Total Waiting Time: {total_waiting_time}"
        )

traci.close()
print("\nSimulation ended.")
