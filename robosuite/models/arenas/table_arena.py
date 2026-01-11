import numpy as np
from random import choice
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion
MATFORYTABLE=["dark","table_ceramic","light-wood", "dark-wood","cherry"]
MATFORWALL=["walls_mat","table_ceramic","light-wood", "dark-wood"]

class TableArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        is_randomize_material = False,
        xml="arenas/table_arena.xml",
    ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_half_size[2]]) + self.table_offset

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]
        self.walls = [
            self.worldbody.find("./geom[@name='wall_leftcorner_visual']"),self.worldbody.find("./geom[@name='wall_rightcorner_visual']"),self.worldbody.find("./geom[@name='wall_rear_visual']")
            ,self.worldbody.find("./geom[@name='wall_front_visual']")
            ,self.worldbody.find("./geom[@name='wall_left_visual']")
            ,self.worldbody.find("./geom[@name='wall_right_visual']")
        ]
        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.configure_location()
        if is_randomize_material:
            self.randomize_material()
    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))

        # If we're not using legs, set their size to 0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # Otherwise, set leg locations appropriately
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # Get z value
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # Set leg position
                leg.set("pos", array_to_string([x, y, -z]))
                # Set leg size
                leg.set("size", array_to_string([0.025, z]))
    #set texture randomly
    #avaliable material: walls_mat,table_legs_metal,table_ceramic,light-wood, dark-wood, 
    # or directly set rgba value, prefer preset in xml file
    def randomize_material(self):
        """Randomizes the material of the table and walls"""
        table_mat = choice(MATFORYTABLE)
        wall_mat = choice(MATFORWALL)
        # table_mat = "novak-wood"
        # floor_mat = "varnished-wood"
        # wall_mat = "wood-tiles"
        self.table_visual.set("material",table_mat)
        for wall in self.walls :
            wall.set("material",wall_mat)
        # self.floor.set("material",floor_mat)
    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset

    def arena_metadata(self):
        metadata = {
            "wall_material": self.walls[0].get("material"),
            "table_material": self.table_visual.get("material"),
        }
        return metadata