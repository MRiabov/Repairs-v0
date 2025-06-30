from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from repairs_components.geometry.b123d_utils import fastener_hole


class TenHoles(EnvSetup):
    "Put 10 fasteners in 10 holes."

    # note: everything is created in mm.

    def desired_state_geom(self) -> Compound:
        with BuildPart() as base_box:
            Box(100, 100, 50)
            with Locations(base_box.filter_by(Axis.Z).sort_by(Axis.Z).last):
                with GridLocations(15, 0, 10, 1):
                    holes = fastener_hole(radius=7, depth=10)
                    fastener = Fastener(True, initial_body_a="base_box@solid")
