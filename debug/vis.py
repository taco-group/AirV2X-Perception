import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D
import numpy as np

def render_pcd_to_image(pcd_path: str, img_path: str = "cloud.png") -> None:
    """Render point cloud to image using matplotlib (safe for headless)."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

# ------------------------------------------------------------------
# EXAMPLE USAGE ----------------------------------------------------
if __name__ == "__main__":
    render_pcd_to_image("bev_ms.pcd", "bev_ms.png")
    render_pcd_to_image("bev.pcd", "bev.png")