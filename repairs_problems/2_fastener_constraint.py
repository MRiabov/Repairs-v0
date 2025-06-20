from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import RepairsSimState
from build123d import *
import genesis as gs
import numpy as np
from repairs_components.geometry.b123d_utils import fastener_hole


class FastenerConstraint(EnvSetup):
    "Simplest env, only for basic debug."

    def starting_state(self):
        with BuildPart() as box_w_hole_1:
            with Locations((0, 0, 5)):
                box = Box(10, 10, 3)

            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):
                with Locations((1, 1, 0)):
                    hole_1, hole_1_loc = fastener_hole(2, 2, joint_name="hole_1")

        with BuildPart() as box_w_hole_2:
            with Locations((0, 0, 5)):
                box = Box(10, 10, 3)

            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):
                with Locations((1, 1, 0)):
                    hole_2, hole_2_loc = fastener_hole(2, 2, joint_name="hole_2")

        box_w_hole_1.part.label = "box_w_hole_1@solid"
        box_w_hole_2.part.label = "box_w_hole_2@solid"

        with Locations((30, 30, 30)):
            fastener = Fastener(
                name="fastener",
                constraint_a_active=True,
                constraint_b_active=True,
                initial_body_a="box_w_hole_1@solid",
                initial_body_b="box_w_hole_2@solid",
            ).bd_geometry()

        return Compound(children=[box_w_hole_1.part, box_w_hole_2.part, fastener])

    def desired_state(
        self, scene: gs.Scene
    ) -> tuple[Compound, RepairsSimState, dict[str, np.ndarray]]:
        with BuildPart() as box_with_holes:
            with Locations((0, 0, 5)):  # pos unchanged
                box = Box(10, 10, 10)

            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):
                with Locations((1, 1, 0)):
                    fastener_hole(2, 2)

        fastener = Fastener().bd_geometry()

        box_with_holes.part.label = "box@solid"
        return (Compound(children=[box_with_holes.part]),)
