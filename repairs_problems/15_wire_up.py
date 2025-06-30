from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from repairs_components.geometry.b123d_utils import fastener_hole
from repairs_components.geometry.connectors.models.europlug import Europlug


class WireUp(EnvSetup):
    "An env with 8 connectors that need to be correctly wired into each hole."

    # note: everything is created in mm.

    def desired_state_geom(self) -> Compound:
        with BuildPart() as elec_panel:
            Box(60, 40, 100)
            with Locations(elec_panel.faces().filter_by(Axis.Y).sort_by(Axis.Y).first):
                hole_grid_locs = GridLocations(0, 30, 1, 4)
                with hole_grid_locs:
                    connector_hole = Box(50, 50, 50, mode=Mode.SUBTRACT)
                for i in range(4):
                    joint = RigidJoint(f"always_{i}", joint_location=hole_grid_locs[i])

            male_geom1, female_geom1, connect_pos1 = Europlug(
                "couple_1@connectors"
            ).bd_geometry(hole_grid_locs.locations[0].to_tuple(), connected=True)
            male_geom2, female_geom2, connect_pos2 = Europlug(
                "couple_2@connectors"
            ).bd_geometry(hole_grid_locs.locations[0].to_tuple(), connected=True)
            male_geom3, female_geom3, connect_pos3 = Europlug(
                "couple_2@connectors"
            ).bd_geometry(hole_grid_locs.locations[0].to_tuple(), connected=True)
            male_geom4, female_geom4, connect_pos4 = Europlug(
                "couple_2@connectors"
            ).bd_geometry(hole_grid_locs.locations[0].to_tuple(), connected=True)

            male_geoms = (male_geom1, male_geom2, male_geom3, male_geom4)
            female_geoms = (female_geom1, female_geom2, female_geom3, female_geom4)
            for i in range(4):
                joint = RigidJoint(f"always_{i}", to_part=female_geoms[i])
                joint.connect_to(other=elec_panel.joints[f"always_{i}"])
                # note: if label name is "always", keep it evne despite perturbations.

            # FIXME: no way to define that a connector would be constrained to other body.

    def linked_groups(self) -> list[tuple[str, ...]]:
        return [()]


show(WireUp().desired_state_geom())
