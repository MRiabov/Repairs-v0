from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *

from repairs_components.geometry.b123d_utils import fastener_hole
from copy import copy

# TODO!!! I don't have randomization (scale) done.


class Fan(EnvSetup):
    "A table from 4 legs and a table top."

    # note: everything is created in mm.

    def desired_state_geom(self) -> Compound:
        fan_locs = ()  # circular locations?
        with BuildPart() as propeller:
            Box(100, 25, 10)
            Box(25, 100, 10)
            with Locations(
                propeller.faces().filter_by(Axis.Z).sort_by(Axis.Z).last.center_location
            ):
                fastener_hole(radius=5, depth=10)
        propeller.moved(Location(translation=(0, 20, 100), rotation=(90, 45, 0)))

        with BuildPart() as base:
            Cylinder(40, 10)
            with Locations((0, 10, 40)):
                Cylinder(15, 80)
            with Locations(Location(translation=(0, 5, 80), rotation=(90, 0, 0))):
                base_top = Cylinder(15, 10, 80)
                with Locations(
                    base_top.faces()
                    .filter_by(Axis.Y)
                    .sort_by(Axis.Y)
                    .first.center_location
                ):
                    fastener_hole(radius=5, depth=7)

        propeller.part.label = "propeller@solid"

        return Compound(children=[fan_locs.part, base.part])


show(Fan().desired_state_geom())
