"""Print the Gym environments registered by the SoftVTBench extension."""

# Launch Isaac Sim before importing Gym environments.
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


# Imports below require the simulator application to be initialized.
import gymnasium as gym

# Import extensions to set up environment tasks
import tac_manip.tasks  # noqa: F401
from prettytable import PrettyTable


def main():
    """Print environments whose entry point or config belongs to ``tac_manip``."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available SoftVTBench Environments"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        config_entry_point = task_spec.kwargs.get("env_cfg_entry_point", "")
        if "tac_manip" not in str(task_spec.entry_point) and "tac_manip" not in str(config_entry_point):
            continue
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, config_entry_point])
        index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
