__author__ = "Simon Wasiela"
__copyright__ = "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
__license__ = "BSD 3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Simon Wasiela"
__email__ = "swasiela@venturi.com"
__status__ = "development"

from scipy.spatial.transform import Rotation as SSTR
from typing import List, Tuple, Union, Dict
import numpy as np
import math
import os

from pxr import UsdLux, Gf, UsdShade

from src.terrain_management.large_scale_terrain.pxr_utils import set_xform_ops, load_material, bind_material
from src.configurations.stellar_engine_confs import SunConf
from src.configurations.environments import EarthConf
from src.environments.base_env import BaseEnv
from src.robots.robot import RobotManager


class EarthController(BaseEnv):
    """
    This class is used to control the lab interactive elements.
    """

    def __init__(
        self,
        earth_settings: EarthConf = None,
        sun_settings: SunConf = None,
        **kwargs,
    ) -> None:
        """
        Initializes the lab controller. This class is used to control the lab interactive elements.
        Including:
            - Sun position, intensity, radius, color.

        Args:
            earth_settings (EarthConf): The settings of the lab.
            sun_settings (SunConf): The settings of the sun object.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(**kwargs)
        self.stage_settings = earth_settings
        self.sun_settings = sun_settings

        self.scene_name = "/Earth"

    def build_scene(self) -> None:
        """
        Builds the scene.
        """

        # Creates an empty xform with the name earth
        earth = self.stage.DefinePrim(self.scene_name, "Xform")
        # Creates the sun
        sun = self.stage.DefinePrim(self.stage_settings.sun_prim, "Xform")
        self._sun_prim = sun.GetPrim()
        self._sun_lux: UsdLux.DistantLight = UsdLux.DistantLight.Define(
            self.stage, os.path.join(self.stage_settings.sun_prim, "sun")
        )
        self._sun_lux.CreateIntensityAttr(self.sun_settings.intensity)
        self._sun_lux.CreateAngleAttr(self.sun_settings.angle)
        self._sun_lux.CreateDiffuseAttr(self.sun_settings.diffuse_multiplier)
        self._sun_lux.CreateSpecularAttr(self.sun_settings.specular_multiplier)
        self._sun_lux.CreateColorAttr(
            Gf.Vec3f(self.sun_settings.color[0], self.sun_settings.color[1], self.sun_settings.color[2])
        )
        self._sun_lux.CreateColorTemperatureAttr(self.sun_settings.temperature)
        x, y, z, w = SSTR.from_euler(
            "xyz", [0, self.sun_settings.elevation, self.sun_settings.azimuth - 90], degrees=True
        ).as_quat()
        set_xform_ops(
            self._sun_lux.GetPrim(), Gf.Vec3d(0, 0, 0), Gf.Quatd(0.5, Gf.Vec3d(0.5, -0.5, -0.5)), Gf.Vec3d(1, 1, 1)
        )
        set_xform_ops(self._sun_prim.GetPrim(), Gf.Vec3d(0, 0, 0), Gf.Quatd(w, Gf.Vec3d(x, y, z)), Gf.Vec3d(1, 1, 1))

        # Load default textures
        self.stage.DefinePrim("/Looks", "Xform")
        load_material("GravelStones", "assets/Textures/GravelStones.mdl")

    def instantiate_scene(self) -> None:
        """
        Instantiates the scene. Applies any operations that need to be done after the scene is built and
        the renderer has been stepped.
        """

        pass

    def reset(self) -> None:
        """
        Resets the environment. Implement the logic to reset the environment.
        """

        pass

    def update(self) -> None:
        """
        Updates the environment.
        """

        pass

    def load(self) -> None:
        """
        Loads the terrain interactive elements in the stage.
        Generates the terrain.
        """
        # Builds the scene
        self.build_scene()

        # Loads the terrain mesh
        self.load_terrain(self.stage_settings.terrain_assets)
       
    def add_robot_manager(self, robotManager: RobotManager) -> None:
        self.robotManager = robotManager

    # ==============================================================================
    # Sun control
    # ==============================================================================

    def set_sun_pose(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """
        Sets the pose of the sun.

        Args:
            position (Tuple[float,float,float]): The position of the sun. In meters. (x,y,z)
            orientation (Tuple[float,float,float,float]): The orientation of the sun. (w,x,y,z)
        """

        w, x, y, z = (orientation[0], orientation[1], orientation[2], orientation[3])
        set_xform_ops(self._sun_prim, orient=Gf.Quatd(w, Gf.Vec3d(x, y, z)))

    def set_sun_intensity(self, intensity: float = 0.0) -> None:
        """
        Sets the intensity of the sun.

        Args:
            intensity (float): The intensity of the projector (arbitrary unit).
        """

        self._sun_lux.GetIntensityAttr().Set(intensity)

    def set_sun_color(self, color: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """
        Sets the color of the projector.

        Args:
            color (Tuple[float,float,float]): The color of the projector. (r,g,b)
        """

        color = Gf.Vec3d(color[0], color[1], color[2])
        self._sun_lux.GetColorAttr().Set(color)

    def set_sun_color_temperature(self, temperature: float = 6500.0) -> None:
        """
        Sets the color temperature of the projector.

        Args:
            temperature (float): The color temperature of the projector in Kelvin.
        """

        self._sun_lux.GetColorTemperatureAttr().Set(temperature)

    def set_sun_angle(self, angle: float = 0.53) -> None:
        """
        Sets the angle of the sun. Larger values make the sun larger, and soften the shadows.

        Args:
            angle (float): The angle of the projector.
        """

        self._sun_lux.GetAngleAttr().Set(angle)

    # ==============================================================================
    # Terrain control
    # ==============================================================================
    def load_terrain(self, terrain_assets: dict = None) -> None:
        """
        Load the terrain from a mesh file

        Args:
            terrain_assets (dict): The dictionnary containing the terrain meshes and material pair.
        """

        assert terrain_assets != None, "No mesh file to load."

        # Get terrain prim path
        terrain_prim_path = "/Earth/Terrain"

        # Loop through all .usd/.mtl pair in the folder
        for mesh_path, mdl_path in terrain_assets:
            # # Define unique prim path
            # mesh_prim_name = os.path.splitext(os.path.basename(mesh_path))[0]

            # Define and reference the USD into the stage
            mesh_prim = self.stage.DefinePrim(terrain_prim_path, "Xform")
            mesh_prim.GetReferences().AddReference(assetPath=mesh_path)

            # Create material prim
            mat_name = os.path.splitext(os.path.basename(mdl_path))[0]
            mat_prim_path = os.path.join("/Looks", mat_name)

            print(mat_prim_path)
    
            # Bind the right material and load it if not already existing
            if not self.stage.GetPrimAtPath(mat_prim_path).IsValid():
                load_material(mat_name, mdl_path)

            bind_material(self.stage, mat_prim_path, mesh_prim.GetPath())

    def deform_derrain(self) -> None:
        """
        Deforms the terrain.
        Args:
            world_poses (np.ndarray): The world poses of the contact points.
            contact_forces (np.ndarray): The contact forces in local frame reported by rigidprimview.
        """
        pass

    def apply_terramechanics(self) -> None:
        pass
