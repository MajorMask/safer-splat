from dataclasses import dataclass, field
from typing import Type, List, Dict
from filterpy.kalman import KalmanFilter
import cvxpy as cp


from nerfbridge.distances import (
    distance_point_ellipsoid,
    batch_point_distance,
    batch_squared_point_distance,
    batch_mahalanobis_distance,
)
from nerfbridge.mesh_utils import create_gs_mesh
from nerfbridge.covariance_utils import quaternion_to_rotation_matrix, compute_cov
from nerfbridge.systems import DoubleIntegrator
from nerfbridge.polytopes_utils import h_rep_minimal

from scipy import sparse
import cv2
import clarabel

import torch
import numpy as np

from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    RGB2SH,
    num_sh_bases,
    random_quat_tensor,
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.cameras import Cameras

import time

from rich.console import Console

CONSOLE = Console(width=80)


@dataclass
class ROSSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: ROSSplatfactoModel)
    depth_seed_pts: int = 5000
    """ Number of points to use for seeding the model from depth per image. """
    seed_with_depth: bool = False
    """ Whether to seed the model from RGBD images. """
    seed_with_pc: bool = True
    """ Whether to seed the model from point clouds. """
    seed_every: int = 4
    """ Seed every k keyframes. """
    min_distance: float = 0.1
    """ Pointcloud minimum seed distance. """
    max_distance: float = 5.0
    """ Pointcloud maximum seed distance. """
    enable_dynamic_ellipsoids: bool = True  # Enable dynamic ellipsoids
    dynamic_update_freq: float = 0.5  # Frequency to update ellipsoids (in seconds)
    max_dynamic_objects: int = 50  # Maximum number of dynamic objects

class ROSSplatfactoModel(SplatfactoModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_img_idx = 0
        self.depth_seed_pts = self.config.depth_seed_pts
        self.seed_with_depth = self.config.seed_with_depth
        self.seed_with_pc = self.config.seed_with_pc
        self.dynamic_objects_kf = []  # List of Kalman filters for dynamic objects

        assert not (
            self.seed_with_depth and self.seed_with_pc
        ), "Cannot seed with both depth and point clouds"

        self.dynamics = DoubleIntegrator(device=self.device, ndim=3)
        # updated in cbf callback in trainer.py :)
        self.viewer_drone_state = None

        self.alpha = 5
        self.beta = 5

        self.locked = False
        self.means_copy = None
        self.scales_copy = None
        self.rots_copy = None

        # For some reason this is not set in the base class
        self.vis_counts = None

        self.seed_every = self.config.seed_every
        self.done_adding = False
        self.render_paper = False

    def populate_modules(self):
        super().populate_modules()
        # Move the initial means far awawy (THIS IS A HACK AND SHOULD BE FIXED)
        self.gauss_params["means"] = self.gauss_params["means"] + 100.0

    def initialize_kalman_filter(self, position, velocity):
        """
        Initialize a Kalman filter for a dynamic object.
        Args:
            position (torch.Tensor): Initial position of the object.
            velocity (torch.Tensor): Initial velocity of the object.
        Returns:
            KalmanFilter: Initialized Kalman filter.
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x[:3] = position.cpu().numpy()  # Initial position
        kf.x[3:] = velocity.cpu().numpy()  # Initial velocity
        kf.F = np.array([[1, 0, 0, 1, 0, 0],  # State transition matrix
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],  # Measurement matrix
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.P *= 10  # Initial uncertainty
        kf.R *= 1  # Measurement noise
        kf.Q *= 0.01  # Process noise
        return kf

    def update_dynamic_objects(self, detected_objects):
        """
        Update the Kalman filters for dynamic objects.
        Args:
            detected_objects (List[Dict]): List of detected objects with positions and velocities.
        """
        # Initialize Kalman filters for new objects
        if len(self.dynamic_objects_kf) == 0:
            for obj in detected_objects:
                kf = self.initialize_kalman_filter(obj["position"], obj["velocity"])
                self.dynamic_objects_kf.append(kf)

        # Update existing Kalman filters
        for kf, obj in zip(self.dynamic_objects_kf, detected_objects):
            kf.predict()
            kf.update(obj["position"].cpu().numpy())

        # Update positions and velocities
        for kf, obj in zip(self.dynamic_objects_kf, detected_objects):
            obj["position"] = torch.tensor(kf.x[:3], device=self.device)
            obj["velocity"] = torch.tensor(kf.x[3:], device=self.device)

    def seed_cb(self, pipeline: Pipeline, optimizers: Optimizers, step: int):
        ds_latest_idx = pipeline.datamanager.train_image_dataloader.buffered_idx

        if ds_latest_idx >= pipeline.datamanager.train_image_dataloader.num_images - 1:
            self.done_adding = True

        if self.seeded_img_idx < ds_latest_idx:
            start_idx = 0 if self.seeded_img_idx == 0 else self.seeded_img_idx + 1
            seed_image_idxs = range(start_idx, ds_latest_idx + 1)
            pre_gaussian_count = self.means.shape[0]
            for idx in seed_image_idxs:
                if idx % self.seed_every != 0:
                    # Only seed every k keyframes
                    continue
                image_data = pipeline.datamanager.train_dataset[idx]
                camera = pipeline.datamanager.train_dataset.cameras[idx]
                with torch.no_grad():
                    self.seed_with_depth = False
                    if self.seed_with_depth:
                        self.seed_from_rgbd(camera, image_data, optimizers)
                    elif self.seed_with_pc:
                        pc = pipeline.datamanager.train_dataset.pc_dict[idx]
                        self.seed_from_pc(camera, pc, optimizers, idx)
                self.seeded_img_idx = idx
            post_gaussian_count = self.means.shape[0]
            diff_gaussians = post_gaussian_count - pre_gaussian_count

            if diff_gaussians == 0:
                return

            if self.xys_grad_norm is not None:
                device = self.xys_grad_norm.device
                self.xys_grad_norm = torch.cat(
                    [self.xys_grad_norm, torch.zeros(diff_gaussians).to(device)]
                )
            if self.max_2Dsize is not None:
                device = self.max_2Dsize.device
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros(diff_gaussians).to(device)]
                )
            if self.vis_counts is not None:
                device = self.vis_counts.device
                self.vis_counts = torch.cat(
                    [self.vis_counts, torch.zeros(diff_gaussians).to(device)]
                )

    def seed_from_rgbd(
        self,
        camera: Cameras,
        image_data: Dict[str, torch.Tensor],
        optimizers: Optimizers,
    ):
        """
        Initialize gaussians at random points in the point cloud from the depth image.

        Means - Initialized to projected points in the depth image.
        Scales - Initialized using k-nearest neighbors approach (same as splatfacto).
        Quats - Initialized to random (same as splatfacto).
        Opacities - Initialized to logit(0.1) (same as splatfacto).
        Features_SH - Initialized to RGB2SH of the points color.
        """
        depth = image_data["depth_image"]
        rgb = image_data["image"]
        H, W, _ = image_data["depth_image"].shape
        depth = torch.nan_to_num(depth, nan=0.0)
        if rgb.device != self.device or depth.device != self.device:
            depth = depth.to(self.device)
            rgb = rgb.to(self.device)

        # Get camera intrinsics and extrinsics
        assert len(camera.shape) == 0
        assert H == camera.image_height.item() and W == camera.image_width.item()
        fx, fy = camera.fx[0].item(), camera.fy[0].item()
        cx, cy = camera.cx[0].item(), camera.cy[0].item()
        c2w = camera.camera_to_worlds.to(self.device)  # (3, 4)
        R = c2w[:3, :3]
        t = c2w[:3, 3].squeeze()

        # Sample pixel indices
        # Could use a confidence map here if available
        nz_row, nz_col = torch.where(depth.squeeze() > 0)
        num_samples = min(self.depth_seed_pts, nz_row.shape[0])
        ind_mask = torch.randperm(nz_row.shape[0])[:num_samples]
        x = nz_col[ind_mask].to(self.device).reshape((-1, 1))
        y = nz_row[ind_mask].to(self.device).reshape((-1, 1))
        rgbs = rgb[y, x, :]  # (num_seed_points, 3)
        rgbs = rgbs.squeeze()

        # Sample depth pixels and project to 3D coordinates (camera relative).
        z = depth[y, x]
        z = z.reshape((-1, 1))  # (num_seed_points, 1)
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        # Flip y and z to switch to opengl coordinate system.
        xyzs = torch.stack([x, -y, -z], dim=-1).squeeze()  # (num_seed_points, 3)

        # Transform camera relative 3D coordinates to world coordinates.
        xyzs = torch.matmul(xyzs, R.T) + t  # (num_seed_points, 3)

        # Initialize scales using 3-nearest neighbors average distance.
        distances, _ = self.k_nearest_sklearn(xyzs, 3)
        distances = torch.from_numpy(distances).to(self.device)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.log(avg_dist.repeat(1, 3))

        # Initialize quats to random.
        quats = random_quat_tensor(self.depth_seed_pts).to(self.device)

        # Initialize SH features to RGB2SH of the points color.
        dim_sh = num_sh_bases(self.config.sh_degree)
        shs = torch.zeros((self.depth_seed_pts, dim_sh, 3)).float().to(self.device)
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(rgbs)
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(rgbs, eps=1e-10)
        features_dc = shs[:, 0, :]
        features_rest = shs[:, 1:, :]

        # Initialize opacities to logit(0.3). This is sort of our opacity prior.
        # Nerfstudio uses a opacity prior of 0.1.
        opacities = torch.logit(0.3 * torch.ones(self.depth_seed_pts, 1)).to(
            self.device
        )

        # Concatenate the new gaussians to the existing ones.
        self.gauss_params["means"] = torch.nn.Parameter(
            torch.cat([self.means.detach(), xyzs], dim=0)
        )
        self.gauss_params["scales"] = torch.nn.Parameter(
            torch.cat([self.scales.detach(), scales], dim=0)
        )
        self.gauss_params["quats"] = torch.nn.Parameter(
            torch.cat([self.quats.detach(), quats], dim=0)
        )
        self.gauss_params["opacities"] = torch.nn.Parameter(
            torch.cat([self.opacities.detach(), opacities], dim=0)
        )
        self.gauss_params["features_dc"] = torch.nn.Parameter(
            torch.cat([self.features_dc.detach(), features_dc], dim=0)
        )
        self.gauss_params["features_rest"] = torch.nn.Parameter(
            torch.cat([self.features_rest.detach(), features_rest], dim=0)
        )


        # Add the new parameters to the optimizer.
        for param_group, new_param in self.get_gaussian_param_groups().items():
            optimizer = optimizers.optimizers[param_group]
            old_param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[old_param]
            added_param_shape = (self.depth_seed_pts, *new_param[0].shape[1:])
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.cat(
                    [
                        param_state["exp_avg"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.cat(
                    [
                        param_state["exp_avg_sq"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )

            del optimizer.state[old_param]
            optimizer.state[new_param[0]] = param_state
            optimizer.param_groups[0]["params"] = new_param
            del old_param

    def seed_from_pc(self, camera: Cameras, pc_data, optimizers: Optimizers, idx):
        """
        Initialize gaussians at random points in the point cloud from the depth image.

        Means - Initialized to projected points in the depth image.
        Scales - Initialized using k-nearest neighbors approach (same as splatfacto).
        Quats - Initialized to random (same as splatfacto).
        Opacities - Initialized to logit(0.1) (same as splatfacto).
        Features_SH - Initialized to RGB2SH of the points color.
        """

        fx, fy = camera.fx[0].item(), camera.fy[0].item()
        cx, cy = camera.cx[0].item(), camera.cy[0].item()
        H = camera.image_height[0].item()
        W = camera.image_width[0].item()
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        c2w = camera.camera_to_worlds.to(self.device)  # (3, 4)
        R = c2w[:3, :3]
        t = c2w[:3, 3].squeeze()

        pc_data = pc_data[~torch.all(pc_data == 0, axis=1)]

        # Project the point cloud to the image plane
        # Remove points not in RGB FOV
        proj, _ = cv2.projectPoints(
            pc_data.numpy(), np.zeros(3), np.zeros(3), K, np.zeros(5)
        )
        proj = torch.from_numpy(proj.squeeze())
        umask = (0 <= proj[:, 0]) & (proj[:, 0] < W)
        vmask = (0 <= proj[:, 1]) & (proj[:, 1] < H)
        in_view = umask & vmask
        pc_data = pc_data[in_view]

        xyzs = pc_data.to(dtype=torch.float32)  # current size (16480, 3) torch tensor

        # Assuming x, y, and z are the individual components from xyzs
        x = xyzs[:, 0]
        y = xyzs[:, 1]
        z = xyzs[:, 2]

        # Flip y and z by negating them and stack them back together
        xyzs = torch.stack([x, -y, -z], dim=-1).squeeze()  # (num_seed_points, 3)

        # Calculate the Euclidean distance to the origin for each point
        distances = torch.norm(xyzs, dim=1)

        # let's filter by distance, remove anything that is too close or too far
        # Define distance thresholds
        min_distance = self.config.min_distance  # Minimum allowable distance
        max_distance = self.config.max_distance  # Maximum allowable distance

        # Create a mask to filter points that are within the allowable distance range
        mask = (distances > min_distance) & (distances < max_distance)
        n_points = mask.sum()
        if n_points == 0:
            return
        elif n_points < self.depth_seed_pts:
            local_n_seed = n_points
        else:
            local_n_seed = self.depth_seed_pts

        # Apply the mask to the point cloud
        xyzs = xyzs[mask]

        # Calculate the range (min, max) of the distances
        min_distance = torch.min(distances)
        max_distance = torch.max(distances)
        indices = torch.randperm(xyzs.size(0))[:local_n_seed]

        # Use the indices to select the downsampled point cloud
        xyzs = xyzs[indices]

        xyzs = xyzs.to(self.device)
        xyzs = torch.matmul(xyzs, R.T) + t  # (num_seed_points, 3)

        # Initialize scales using 3-nearest neighbors average distance.
        ## Spawn them all as the same size gaussian for now
        scales = torch.ones(local_n_seed, 3).to(self.device) * -4.5

        # Initialize quats to random.
        quats = random_quat_tensor(local_n_seed).to(self.device)

        # Initialize SH features to RGB2SH of the points color.
        dim_sh = num_sh_bases(self.config.sh_degree)
        # lets make the shs init to random
        shs = torch.rand((local_n_seed, dim_sh, 3)).float().to(self.device)
        features_dc = shs[:, 0, :]
        features_rest = shs[:, 1:, :]

        # Initialize opacities to logit(0.3). This is sort of our opacity prior.
        # Nerfstudio uses a opacity prior of 0.1.
        opacities = torch.logit(0.3 * torch.ones(local_n_seed, 1)).to(self.device)

        # Concatenate the new gaussians to the existing ones.
        self.gauss_params["means"] = torch.nn.Parameter(
            torch.cat([self.means.detach(), xyzs], dim=0)
        )
        self.gauss_params["scales"] = torch.nn.Parameter(
            torch.cat([self.scales.detach(), scales], dim=0)
        )
        self.gauss_params["quats"] = torch.nn.Parameter(
            torch.cat([self.quats.detach(), quats], dim=0)
        )
        self.gauss_params["opacities"] = torch.nn.Parameter(
            torch.cat([self.opacities.detach(), opacities], dim=0)
        )
        self.gauss_params["features_dc"] = torch.nn.Parameter(
            torch.cat([self.features_dc.detach(), features_dc], dim=0)
        )
        self.gauss_params["features_rest"] = torch.nn.Parameter(
            torch.cat([self.features_rest.detach(), features_rest], dim=0)
        )

        # Add the new parameters to the optimizer.
        for param_group, new_param in self.get_gaussian_param_groups().items():
            optimizer = optimizers.optimizers[param_group]
            old_param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[old_param]
            added_param_shape = (local_n_seed, *new_param[0].shape[1:])
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.cat(
                    [
                        param_state["exp_avg"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.cat(
                    [
                        param_state["exp_avg_sq"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )

            del optimizer.state[old_param]
            optimizer.state[new_param[0]] = param_state
            optimizer.param_groups[0]["params"] = new_param
            del old_param
    
    """
    Detect moving objects using depth and pose data.
    Args:
        depth_data (torch.Tensor): Depth image.
        pose_data (torch.Tensor): Current pose of the camera.
        prev_pose_data (torch.Tensor): Previous pose of the camera.
    Returns:
        List[Dict]: List of detected moving objects with positions and velocities.
    """
    def detect_moving_objects(self, depth_data, pose_data, prev_pose_data):

        # Compute relative motion between current and previous poses
        relative_motion = pose_data[:3, 3] - prev_pose_data[:3, 3]

        # Extract 3D points from depth data
        nz_row, nz_col = torch.where(depth_data > 0)
        z = depth_data[nz_row, nz_col]
        x = (nz_col - self.config.cx) * z / self.config.fx
        y = (nz_row - self.config.cy) * z / self.config.fy
        points_3d = torch.stack([x, y, z], dim=-1)

        # Transform points to world coordinates
        points_world = torch.matmul(points_3d, pose_data[:3, :3].T) + pose_data[:3, 3]

        # Compute velocities (approximation using relative motion)
        velocities = relative_motion.unsqueeze(0).repeat(points_world.shape[0], 1)

        # Filter dynamic objects based on velocity threshold
        velocity_magnitude = torch.norm(velocities, dim=1)
        dynamic_mask = velocity_magnitude > self.config.dynamic_velocity_threshold

        dynamic_objects = []
        for point, velocity in zip(points_world[dynamic_mask], velocities[dynamic_mask]):
            dynamic_objects.append({"position": point, "velocity": velocity})

        return dynamic_objects

    def cbf(self, u_des, state):
        # lets access the gaussian parameters
  # Dynamically adjust alpha and beta based on the system's state
        velocity_norm = torch.norm(state[3:6])
        self.alpha = 5 + 2 * velocity_norm  # Example: Increase alpha with velocity
        self.beta = 5 + 1 * velocity_norm  # Example: Increase beta with velocity

        # self.gauss_params["means"]
        tnow = time.time()
        h, grad_h, hes_h, info = self.query_distance(state, radius=0.25)
        CONSOLE.print(f"[blue] QUERY: {time.time() - tnow:0.5f}")

        # print("min dist ", torch.min(h))
        min_dist = torch.min(h)

        # publish h info
        tnow = time.time()
        with torch.no_grad():
            A, l, P, q = self.get_QP_matrices(
                state, u_des, h, grad_h, hes_h, minimal=True
            )
        #     tnow = time.time()
        CONSOLE.print(f"[blue] GETQP: {time.time() - tnow:0.5f}")

        tnow = time.time()
        u_out, success_flag = self.optimize_QP_clarabel(A, l, P, q)
        CONSOLE.print(f"[blue] SOLVEQP: {time.time() - tnow:0.5f}")

        #     print('Time to solve QP:', time.time() - tnow)

        #     self.solver_success = success_flag

        if success_flag:
            # return the optimal control
            # u_out = torch.tensor(u_out).to(device=u_des.device, dtype=torch.float32)
            u_out = np.array(u_out)
        else:
            # if not successful, just return the desired control but raise a warning
            print("Solver failed. Returning desired control.")
            u_out = u_des

        return u_out, min_dist

    def clone_cb(self, step):
        if not self.locked:
            # torch.cuda.synchronize()
            self.means_copy = self.means.detach().clone()
            self.rots_copy = self.quats.detach().clone()
            self.scales_copy = self.scales.detach().clone()
            # torch.cuda.synchronize()

        if step % 15 == 0:
            CONSOLE.print(
                f"[yellow] Number of Gaussians: {self.means_copy.shape[0]} {self.done_adding}"
            )

    def check_param_dims(self):
        Ng = self.means_copy.shape[0]
        return (Ng == self.rots_copy.shape[0]) and (Ng == self.scales_copy.shape[0])

    # NOTE: Need to provide the robot radius OR the robot R and S matrices
    def query_distance(
        self, x, distance_type=None, radius=0.0, R_robot=None, S_robot=None, epsilon=0.0
   , dynamic_objects=None ):
        # Queries varieties of distance from x to the GSplat.
        if self.means_copy is None:
            return

        self.locked = True

        if not self.check_param_dims():
            CONSOLE.print(f"[bold orange]Gaussian params not the same size!")
            CONSOLE.print(f"[bold orange]Attempting re-clone!")
            self.clone_cb(0)
            if self.check_param_dims():
                pass
            else:
                CONSOLE.rule(f"[bold orange]Re-clone FAILED!")
                self.locked = False
                return None, None, None
        # FISHY
        if dynamic_objects:
            for obj in dynamic_objects:
                position = obj["position"]
                velocity = obj["velocity"]

                # Add Gaussian parameters for the dynamic object
                self.gauss_params["means"] = torch.cat([self.gauss_params["means"], position.unsqueeze(0)], dim=0)
                self.gauss_params["scales"] = torch.cat([self.gauss_params["scales"], torch.log(torch.tensor([0.5, 0.5, 0.5]))], dim=0)
                self.gauss_params["quats"] = torch.cat([self.gauss_params["quats"], random_quat_tensor(1).to(self.device)], dim=0)
                self.gauss_params["opacities"] = torch.cat([self.gauss_params["opacities"], torch.logit(torch.tensor([0.3]))], dim=0)


        means = self.means_copy
        rots = self.rots_copy
        scales = self.scales_copy

        scales = torch.exp(scales)
        #
        # print("cloning time",time.time() - start_time)

        cov_inv = compute_cov(rots, 1 / scales)
        covs = compute_cov(rots, scales)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if distance_type == "ball-to-ball":
            ball_radius = torch.max(scales, dim=-1)[0]
            dist, grad, hess = batch_point_distance(x[..., :3], means)

            h = dist - (ball_radius + radius + epsilon)
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == "ball-to-ball-squared":
            ball_radius = torch.max(scales, dim=-1)[0]
            squared_dist, grad, hess = batch_squared_point_distance(x[..., :3], means)

            h = squared_dist - (ball_radius + radius + epsilon) ** 2
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == "mahalanobis":
            maha_dist, grad, hess = batch_mahalanobis_distance(x, means, cov_inv)

            h = maha_dist - 1.0
            grad_h = grad
            hess_h = hess

            info = None

        elif (distance_type is None) or (distance_type == "ball-to-ellipsoid"):
            # Queries the min Euclidian distance from point to ellipsoid
            # Rotate point into the ellipsoid frame

            # Convert rotations from quaternions to rotation matrices
            rots = quaternion_to_rotation_matrix(rots)

            # Sort the scales in descending order as required by the solver
            sorted_output = torch.sort(scales, dim=-1, descending=True)
            sorted_scales, sorted_inds = sorted_output[0], sorted_output[1]

            # NOTE:!!! IMPORTANT!!! When we sort, we need to change the rotation matrices accordingly
            rots = torch.gather(rots, 2, sorted_inds[..., None, :].expand_as(rots))

            # Translate robot w.r.t ellipsoid mean, then rotate point into ellipsoid aligned frame
            x_local_frame = (
                torch.bmm(
                    torch.transpose(rots, 1, 2), (x[..., :3] - means).unsqueeze(-1)
                ).squeeze()
                + 1e-8
            )

            # The solver requires the point to be in the first octant. Calculate the sign of the point and flip the point.
            flip = torch.sign(x_local_frame)
            x_local_frame = torch.abs(x_local_frame)

            # solve for the distance in the local frame
            dist, _, hess, yhat = distance_point_ellipsoid(
                sorted_scales + 1e-8, x_local_frame
            )

            # flip, rotate, and translate the closest point back to the global frame
            y = torch.bmm(rots, (flip * yhat).unsqueeze(-1)).squeeze(-1) + means

            # Calculate cbf
            phi = torch.sign(
                torch.sum((1.0 / sorted_scales) ** 2 * (x_local_frame**2), dim=-1)
                - 1.0
            )

            h = phi * dist - (radius + epsilon) ** 2

            # Compute gradient in world frame.
            grad_h = 2 * phi[..., None] * (x[..., :3] - y)

            # Mutliple Hessian by phi
            hess_h = phi[..., None, None] * hess

            info = {"y": y, "phi": phi}

        else:
            raise ValueError(
                "Distance type not recognized. Please provide a valid distance type."
            )

        self.locked = False
        return h, grad_h, hess_h, info

    # TODO: This function assumes relative degree 2, we should make it account for single-integrator dynamics too.
    def get_QP_matrices(self, x, u_des, h, grad_h, hes_h, dynamic_objects = None, minimal=True):
        # Computes the A and b matrices for the QP A u <= b
        tnow = time.time()
        # h, grad_h, hes_h, info = self.gsplat.query_distance(x[..., :3], radius=self.radius)       # can pass in an optional argument for a radius
        # print('Time to query distance:', time.time() - tnow)
        if dynamic_objects:
            for obj in dynamic_objects:
                position = obj["position"]
                velocity = obj["velocity"]

            # Compute dynamic constraints
                dynamic_h, dynamic_grad_h, dynamic_hess_h = self.compute_dynamic_constraints(position, velocity)

                # Append dynamic constraints to QP matrices
                h = torch.cat([h, dynamic_h], dim=0)
                grad_h = torch.cat([grad_h, dynamic_grad_h], dim=0)
                hes_h = torch.cat([hes_h, dynamic_hess_h], dim=0)
        h = h.unsqueeze(-1)
        grad_h = torch.cat(
            (grad_h, torch.zeros(h.shape[0], 3).to(grad_h.device)), dim=-1
        )
        # add zeros to the right of hes_h
        hes_h = torch.cat(
            (hes_h, torch.zeros(h.shape[0], 3, 3).to(hes_h.device)), dim=2
        )
        hes_h = torch.cat(
            (hes_h, torch.zeros(h.shape[0], 3, 6).to(hes_h.device)), dim=1
        )

        f, g, df = self.dynamics.system(x)

        f = f.unsqueeze(-1)

        f = f.to(grad_h.device)
        g = g.to(f.device)
        df = df.to(f.device)

        # Extended barrier function
        lfh = torch.matmul(grad_h, f).squeeze()

        lflfh = (
            torch.matmul(f.T, torch.matmul(hes_h, f)).squeeze()
            + torch.matmul(grad_h, torch.matmul(df, f)).squeeze()
        )
        lglfh = (
            torch.matmul(g.T, torch.matmul(hes_h, f)).squeeze()
            + torch.matmul(grad_h, torch.matmul(df, g)).squeeze()
        )

        # our full constraint is
        # lflfh + lglfh * u + alpha(lfh) + beta(lfh + alpha(h)) <= 0
        l = -lflfh - self.alpha * lfh - self.beta * (lfh + self.alpha * (h.squeeze()))

        A = lglfh[None]  # 1 x 6

        P = np.eye(3)
        q = -1 * u_des.cpu().numpy()

        A = A.cpu().numpy().squeeze()
        l = l.cpu().numpy()

        # We want to solve for a minimal set of constraints in the Polytope
        # Normalize
        norms = np.linalg.norm(A, axis=-1, keepdims=True)
        A = -A / norms
        l = -l / norms.squeeze()

        # Try to find minimal set of polytopes
        tnow = time.time()
        if minimal:
            # We know that the collision-less ellipsoids have CBF constraints that contain the origin.
            # For those that are in collision, we don't know if the origin is in the polytope and we should
            # avoid trying to solve an optimization problem to find the interior. Because these constraints
            # are relatively few, we can just put them in the QP as is.
            collisionless = (h.cpu().numpy() > 0).squeeze()

            collisionless_A = A[collisionless]
            collisionless_l = l[collisionless]

            collision_A = A[~collisionless]
            collision_l = l[~collisionless]

            # print('Is Robot not in Collision?: ', np.all(collisionless), 'Number of collisions:', np.sum(~collisionless))

            try:
                try:
                    # If the robot is safe, the origin should be solution (u = -(alpha + beta) v)
                    feasible_pt = -(self.alpha + self.beta) * x[..., 3:6].cpu().numpy()
                    Aminimal, lminimal = h_rep_minimal(
                        collisionless_A, collisionless_l, feasible_pt
                    )
                except:
                    # print('The origin is not a feasible point. Resorting to solving Chebyshev center for an interior point.')
                    # Find interior point through Chebyshev center
                    # feasible_pt = find_interior(A, l)
                    # Aminimal, lminimal = h_rep_minimal(A, l, feasible_pt)
                    raise ValueError(
                        "Failed to find an interior point for the minimal polytope."
                    )

                # print('Reduction in polytope size:', 1 - Aminimal.shape[0] / A.shape[0], 'Final polytope size:', Aminimal.shape[0])
                A, l = np.concatenate([Aminimal, collision_A], axis=0), np.concatenate(
                    [lminimal, collision_l], axis=0
                )
            except:
                # print('Failed to compute minimal polytope. Keeping all constraints.')
                pass
        # print('Time to compute minimal polytope:', time.time() - tnow)

        return A, l, P, q

    # # TODO: We need to make sure that we transform the u_out into the world frame from the ellipsoid frame for ellipsoid-ellipsoid
    # def solve_QP(self, x, u_des):
    #     A, l, P, q = self.get_QP_matrices(x, u_des, minimal=True)

    #     tnow = time.time()
    #     u_out, success_flag = self.optimize_QP_clarabel(A, l, P, q)
    #     print('Time to solve QP:', time.time() - tnow)

    #     self.solver_success = success_flag

    #     if success_flag:
    #         # return the optimal control
    #         u_out = torch.tensor(u_out).to(device=u_des.device, dtype=torch.float32)
    #     else:
    #         # if not successful, just return the desired control but raise a warning
    #         print('Solver failed. Returning desired control.')
    #         u_out = u_des

    #     #TODO: We might want to try CVXPY as a backup solver if Clarabel fails.

    #     return u_out

    def compute_dynamic_constraints(self, position, velocity):
        """
        Compute constraints for a dynamic object.
        Args:
            position (torch.Tensor): Position of the dynamic object.
            velocity (torch.Tensor): Velocity of the dynamic object.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: h, grad_h, hess_h for the dynamic object.
        """
        # Example: Use a simple distance-based constraint
        h = torch.norm(position - self.robot_position) - self.dynamic_radius
        grad_h = 2 * (position - self.robot_position)
        hess_h = 2 * torch.eye(3)

        return h.unsqueeze(0), grad_h.unsqueeze(0), hess_h.unsqueeze(0)

    # Clarabel is a more robust, faster solver
    def optimize_QP_clarabel(self, A, l, P, q):
        n_constraints = A.shape[0]

        # Setup workspace
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(
            P, q, A, l, [clarabel.NonnegativeConeT(n_constraints)], settings
        )
        sol = solver.solve()

    # Check solver status
        if str(sol.status) != "Solved":
            print("Clarabel failed. Falling back to CVXPY.")
            # Fallback to CVXPY
            u = cp.Variable(3)
            objective = cp.Minimize(0.5 * cp.quad_form(u, P) + q.T @ u)
            constraints = [A @ u <= l]
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve()
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    return u.value, True
                else:
                    return None, False
            except Exception as e:
                print(f"CVXPY failed: {e}")
                return None, False
        else:
            return sol.x, True

# FISHY
    def training_step(self, pipeline, optimizers, step):
    
        # Detect moving objects
        dynamic_objects = self.detect_moving_objects(
            depth_data=pipeline.datamanager.train_dataset.depth_image,
            pose_data=pipeline.datamanager.train_dataset.pose,
            prev_pose_data=pipeline.datamanager.train_dataset.prev_pose,
        )

        # Query distances and add dynamic ellipsoids
        h, grad_h, hess_h, info = self.query_distance(
            x=self.robot_state, dynamic_objects=dynamic_objects)

        # Update QP constraints
        A, l, P, q = self.get_QP_matrices(
            x=self.robot_state, u_des=self.desired_control, h=h, grad_h=grad_h, hes_h=hess_h, dynamic_objects=dynamic_objects
        )

        # Solve QP and update control
        u_out, success_flag = self.optimize_QP_clarabel(A, l, P, q)
        self.update_control(u_out)
   
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs_base = super().get_training_callbacks(training_callback_attributes)

        if self.config.seed_with_pc or self.config.seed_with_depth:
            cb_seed = TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.seed_cb,
                args=[
                    training_callback_attributes.pipeline,
                    training_callback_attributes.optimizers,
                ],
            )
        pose_buffer = TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            training_callback_attributes.pipeline.datamanager.train_image_dataloader.update_dataset_from_buffer,
            args=[],
        )
        copy_cb = TrainingCallback(
            [TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.clone_cb, args=[]
        )
        return [cb_seed, pose_buffer, copy_cb] + cbs_base
