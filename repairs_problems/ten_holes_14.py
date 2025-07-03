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
            Box(20, 20, 20)
            with Locations(base_box.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                grid_locs = GridLocations(1.5, 0, 10, 1)
                with grid_locs:
                    holes, hole_locs = fastener_hole(radius=7, depth=10)

        fastener, collision_detection_position = Fastener(
            False, initial_body_a="base_box@solid"
        ).bd_geometry()
        fasteners = [fastener.moved(loc) for loc in grid_locs.locations]

        return Compound(children=[base_box.part, *fasteners])


if __name__ == "__main__":
    env = TenHoles()
    show(env.desired_state_geom())
