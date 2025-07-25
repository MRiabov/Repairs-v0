from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import RepairsSimState
from build123d import *
import genesis as gs
import numpy as np
from repairs_components.geometry.b123d_utils import fastener_hole


class FastenersIntoBox(EnvSetup):
    "Simplest env, only for basic debug."

    def desired_state_geom(self) -> Compound:
        with BuildPart() as box_with_holes:
            with Locations((0, 0, 5)):  # pos unchanged
                box = Box(10, 10, 10)

            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):
                with Locations((1, 1, 0)):
                    fastener_hole(2, 2)

        fastener = Fastener(initial_body_a="box_with_holes@solid").bd_geometry()

        box_with_holes.part.label = "box@solid"
        return (Compound(children=[box_with_holes.part]),)
