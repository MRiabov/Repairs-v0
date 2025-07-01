from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from repairs_components.geometry.b123d_utils import fastener_hole


class Chair(EnvSetup):
    "A toy bike to assemble."

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

        with BuildPart() as sitting_plate:
            plate_size_xy = 50 * 2 + 10 * 2
            plate = Box(plate_size_xy, plate_size_xy, 15)
            with Locations(plate.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                with GridLocations(100, 100, 2, 2):
                    leg_holes = fastener_hole(radius=5, depth=15)
            with Locations(
                plate.faces().filter_by(Axis.Y).sort_by(Axis.Y).first.center_location
            ):  # note: not sure, maybe this needs to be relocated to -40,0,0. does grid spawn in center?...
                with GridLocations(80, 0, 2, 1):
                    rear_holes = fastener_hole(radius=5, depth=7)
        with BuildPart() as rear_stick:
            block = Box(10, 10, 80)
            with Locations(
                block.faces().filter_by(Axis.Y).sort_by(Axis.Y).first.center_location
            ):
                with Locations((0, -30, 0)):
                    rear_stick_low_holes = fastener_hole(radius=5, depth=10)

            with Locations(
                block.faces().filter_by(Axis.X).sort_by(Axis.X).last.center_location
            ):
                with Locations((0, 20, 0)):
                    rear_stick_top_holes = fastener_hole(radius=5, depth=10)
        with BuildPart() as back:
            back_plate = Box(80 - (10 * 2 / 2), 10, 30)
            left_face, right_face = back_plate.faces().sort_by(Axis.X)[[0, -1]]
            with Locations(left_face.center_location):
                left_hole = fastener_hole(radius=5, depth=10)
            with Locations(right_face.center_location):
                right_hole = fastener_hole(radius=5, depth=10)

        rear_stick = rear_stick.move((-40, 50 + 15 / 2, 100 + 80 / 2))
        rear_stick2 = rear_stick.moved((80, 0, 0))
        back.move(0, -(50 + 10 / 2), 100 + 60)  # or so Z

        # TODO fasteners...

        for leg_ in (leg, leg2, leg3, leg4):
            leg_.part.label = "leg@solid"

        rear_stick.part.label = "rear_stick@solid"
        rear_stick2.part.label = "rear_stick@solid"
        back.part.label = "chair_back@solid"
        sitting_plate.part.label = "sitting_plate@solid"

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
