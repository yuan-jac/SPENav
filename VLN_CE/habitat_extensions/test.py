# test_habitat_sim.py
import traceback
try:
    import habitat_sim
    print("habitat_sim imported:", habitat_sim.__version__ if hasattr(habitat_sim, "__version__") else "unknown")
except Exception:
    print("IMPORT ERROR")
    traceback.print_exc()
    raise SystemExit(1)

from habitat_sim import Simulator, SimulatorConfiguration, AgentConfiguration, SensorSpec
try:
    sim_cfg = SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    # try enabling GPU rendering flag if available
    try:
        sim_cfg.gpu_rendering = True
    except Exception:
        pass

    sim = Simulator(sim_cfg)
    print("Simulator created successfully")
    # optional: try load small scene (adjust path)
    # sim.scene_manager.load_scene("/path/to/your/scene.glb")
except Exception:
    print("SIM INIT/SCENE LOAD ERROR")
    traceback.print_exc()
