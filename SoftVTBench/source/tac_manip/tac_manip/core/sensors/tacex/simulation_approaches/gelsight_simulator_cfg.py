from isaaclab.utils import configclass

"""Configuration for a tactile RGB simulation with Taxim."""


@configclass
class GelSightSimulatorCfg:
    """Parent Class for Simulation Approach Cfg classes.

    Basically, only `simulation_approach_class` is important (right now at least).
    It could very well be that this class is pretty much useless/overkill.
    """

    simulation_approach_class: type = None
    """"""
    device: str = "cuda"
