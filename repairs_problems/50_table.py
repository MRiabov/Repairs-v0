from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from repairs_components.geometry.b123d_utils import fastener_hole


class Table(EnvSetup):
    "A table from 4 legs and a table top."

    # note: everything is created in mm.

    def desired_state_geom(self) -> Compound:
        with BuildPart() as leg:
            with Locations((-50, -50, 50)):
                leg_base = Box(20, 20, 100)
                with Locations(leg_base.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                    top_hole = fastener_hole(radius=5, depth=7)

        leg2 = leg.moved(100, 0, 0)
        leg3 = leg.moved(0, 100, 0)
        leg4 = leg.moved(100, 100, 0)

        with BuildPart() as table_top:
            plate_size_xy = 50 * 2 + 10 * 2
            plate = Box(plate_size_xy, plate_size_xy, 15)
            with Locations(plate.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                with GridLocations(100, 100, 2, 2):
                    leg_holes = fastener_hole(radius=5, depth=15)

        # TODO fasteners...

        for leg_ in (leg, leg2, leg3, leg4):
            leg_.part.label = "leg@solid"

        back.part.label = "chair_back@solid"
        table_top.part.label = "table_top@solid"

        return Compound(
            children=[
                leg.part,
                leg2.part,
                leg3.part,
                leg4.part,
                sitting_plate.part,
                rear_stick.part,
                rear_stick2.part,
                back.part,
            ]
        )


show(Chair().desired_state_geom())
