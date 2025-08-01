import math

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from shapely.geometry import MultiPoint


def load_camera_data(camera_files, preload=True):
    """
    Args:
        camera_files: list,
            store camera path
        shape : tuple
            (width, height), resize the image, and overcoming the lazy loading.
    Returns:
        camera_data_list: list,
            list of Image, RGB order
    """
    camera_data_list = []
    for camera_file in camera_files:
        camera_data = Image.open(camera_file)
        if preload:
            camera_data = camera_data.copy()
        camera_data_list.append(camera_data)
    return camera_data_list


def sample_augmentation(data_aug_conf, is_train):
    """
    https://github.com/nv-tlabs/lift-splat-shoot/blob/d74598cb51101e2143097ab270726a561f81f8fd/src/data.py#L96
    """
    H, W = data_aug_conf['H'], data_aug_conf['W']
    fH, fW = data_aug_conf['final_dim']
    if is_train:
        resize = np.random.uniform(*data_aug_conf['resize_lim'])
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH) # [x_start, y_start, x_end, y_end]
        flip = False
        if data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
            flip = True
        rotate = np.random.uniform(*data_aug_conf['rot_lim'])
    else:
        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
    return resize, resize_dims, crop, flip, rotate



def img_transform(imgs, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    imgs_output = []
    for img in imgs:
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        imgs_output.append(img)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])

    if flip: 
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2 # [x_start, y_start, x_end, y_end]
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return imgs_output, post_rot, post_tran


def get_rot(h):
    return torch.Tensor(
        [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
    )


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def denormalize_tensor(
    input,
    mean=torch.as_tensor([0.485, 0.456, 0.406]),
    std=torch.as_tensor([0.229, 0.224, 0.225]),
):
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    mean_inv = mean_inv.view(3, 1, 1).to(input.device)
    std_inv = std_inv.view(3, 1, 1).to(input.device)
    input = (input - mean_inv) / std_inv
    return input


denormalize_img = torchvision.transforms.Compose(
    (
        NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
    )
)


normalize_img = torchvision.transforms.Compose(
    (
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    )
)

def decode_depth_carla(depth_map_ori, to_PIL=True): 
    
    """
    Decode depth map from CARLA
    Args:
        depth_map (np.ndarray): Depth map in PIL format
        size (tuple): Size to resize the depth map to (width, height)
    Returns:
        depth_map (np.ndarray): Decoded depth map in meters
    """
    depth_map = np.array(depth_map_ori).astype(np.uint32)
    depth_map = depth_map[:, :, 0] + depth_map[:, :, 1] * 256 + depth_map[:, :, 2] * 256 * 256
    depth_map = depth_map.astype(np.float64)  / (256 * 256 * 256 - 1) * 1000
    # import matplotlib.pyplot as plt
    # plt.imsave("debug/depth_map.png", depth_map, cmap='gray')
    # import pdb; pdb.set_trace()
    if to_PIL:
        depth_scaled = np.clip(depth_map * 65535 / 1000, 0, 65535).astype(np.uint16)
        pil_depth = Image.fromarray(depth_scaled, mode='I;16')
        return pil_depth
    depth_map = torch.from_numpy(depth_map)
    return depth_map


# def decode_depth_carla(depth_map, to_PIL=True): 
    
#     """
#     Decode depth map from CARLA
#     Args:
#         depth_map (np.ndarray): Depth map in PIL format
#         size (tuple): Size to resize the depth map to (width, height)
#     Returns:
#         depth_map (np.ndarray): Decoded depth map in meters
#     """
#     depth_map = np.array(depth_map)
#     depth_map = depth_map[:, :, 1] + depth_map[:, :, 2] * 256 + depth_map[:, :, 0] * 256 * 256
#     # Normalize to get values in meters (1.0/256.0 factor is provided in CARLA docs)
#     depth_map = depth_map.astype(np.float32) * (1.0 / 256.0)
#     if to_PIL:
#         depth_scaled = np.clip(depth_map * 65.535, 0, 65535).astype(np.uint16)
#         pil_depth = Image.fromarray(depth_scaled, mode='I;16')
#         return pil_depth
#     depth_map = torch.from_numpy(depth_map)
#     return depth_map



def pil_depth_to_tensor(pil_depth):
    
    """
    Convert a PIL image of depth to a tensor. Should to be used in pair with decode_depth_carla
    Args:
        pil_depth (PIL.Image): PIL image of depth in 16-bit format.
    Returns:
        depth_tensor (torch.Tensor): Tensor of depth values in meters.
    """
    
    if pil_depth.mode != 'I;16':
        raise ValueError("Input must be single channel uint16 (mode='I;16')")
    depth_scaled = np.array(pil_depth, dtype=np.float32)
    
    depth_meters = depth_scaled * 1000 / 65535.0
    
    depth_tensor = torch.from_numpy(depth_meters)
    
    return depth_tensor


# def pil_depth_to_tensor(pil_depth):
    
#     """
#     Convert a PIL image of depth to a tensor. Should to be used in pair with decode_depth_carla
#     Args:
#         pil_depth (PIL.Image): PIL image of depth in 16-bit format.
#     Returns:
#         depth_tensor (torch.Tensor): Tensor of depth values in meters.
#     """
    
#     if pil_depth.mode != 'I;16':
#         raise ValueError("Input must be single channel uint16 (mode='I;16')")
#     depth_scaled = np.array(pil_depth, dtype=np.float32)
    
#     depth_meters = depth_scaled / 65.535
    
#     depth_tensor = torch.from_numpy(depth_meters)
#     import pdb; pdb.set_trace()
#     return depth_tensor

    
    
img_to_tensor = torchvision.transforms.ToTensor()  # [0,255] -> [0,1]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [int((row[1] - row[0]) / row[2] + 0.5) for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=True):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        # mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        # indices[mask] = num_bins
        indices[indices < 0] = 0
        indices[indices >= num_bins] = num_bins - 1
        indices[~torch.isfinite(indices)] = num_bins - 1

        # Convert to integer
        indices = indices.type(torch.int64)
        return indices, None
    else:
        # mask indices outside of bounds
        mask = (indices < 0) | (indices >= num_bins) | (~torch.isfinite(indices))
        indices[indices < 0] = 0
        indices[indices >= num_bins] = num_bins - 1
        indices[~torch.isfinite(indices)] = num_bins - 1

        # Convert to integer
        indices = indices.type(torch.int64)
        return indices, ~mask


def depth_discretization(depth_min, depth_max, num_bins, mode):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        depth_discre = depth_min + bin_size * np.arange(num_bins)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depth_discre = (
            depth_min
            + bin_size * (np.arange(num_bins) * np.arange(1, 1 + num_bins)) / 2
        )
    else:
        raise NotImplementedError
    return depth_discre


def indices_to_depth(indices, depth_min, depth_max, num_bins, mode):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        depth = indices * bin_size + depth_min
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depth = depth_min + bin_size * (indices * (indices + 1)) / 2
    else:
        raise NotImplementedError
    return depth


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def get_2d_bounding_box(cords):
    """
    transform the 3D bounding box to 2D
    :param cords: <3, 8> the first channel: x, y, z; the second channel is the points amount
    :return <4, > 2D bounding box (x, y, w, h)
    """
    x_min = min(cords[0])
    x_max = max(cords[0])
    y_min = min(cords[1])
    y_max = max(cords[1])
    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    return bbox


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    # print('r1, r2, r3: ', r1, r2, r3)
    return min(r1, r2, r3)


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)

    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap


def draw_gaussian_mask(heatmap, gt_box2d, gt_box2d_mask):
    MAX_H, MAX_W = heatmap.shape
    for gt_box in gt_box2d[gt_box2d_mask]:
        bbox = get_2d_bounding_box(gt_box.T)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, MAX_W - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, MAX_H - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        ct_int = ct.astype(np.int32)
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            draw_msra_gaussian(heatmap, ct_int, radius)
    return heatmap


def draw_bin_mask(heatmap, gt_box2d, gt_box2d_mask):
    for gt_box in gt_box2d[gt_box2d_mask]:
        poly = MultiPoint(gt_box).convex_hull
        cv2.fillConvexPoly(
            heatmap, np.array(list(zip(*poly.exterior.coords.xy)), dtype=np.int32), 1
        )
    return heatmap


def coord_3d_to_2d(
    gt_box3d,
    int_matrix,
    ext_matrix,
    mask="bin",
    image_H=600,
    image_W=800,
    image=None,
    idx=None,
):
    """
    Projects XYZ points onto the canvas and returns the projected canvas
    coordinates.

    Args:
        gt_box3d : np.ndarray
            shape (N, 8, 3). point coord in world (LiDAR) coordinate.
        int_matrix : np.ndarray
            shape (4, 4)
        ext_matrix : np.ndarray
            shape (4, 4), T_wc, transform point in camera coord to world coord.

    Returns:
        gt_box2d : np.ndarray
            shape (N, 8, 2). pixel coord (u, v) in the image. You may want to flip them for image data indexing.
        gt_box2d_mask : np.ndarray (bool)
            shape (N,). If false, this box is out of image boundary
        fg_mask : np.ndarray
            shape (image_H, image_W), 1 means foreground, 0 means background
    """
    N = gt_box3d.shape[0]
    xyz = gt_box3d.reshape(-1, 3)  # (N*8, 3)

    xyz_hom = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
    )

    ext_matrix = np.linalg.inv(ext_matrix)[:3, :4]
    img_pts = (int_matrix @ ext_matrix @ xyz_hom.T).T

    depth = img_pts[:, 2]
    uv = img_pts[:, :2] / depth[:, None]
    uv_int = uv.round().astype(np.int32)  # [N*8, 2]

    # o--------> u
    # |
    # |
    # |
    # v v

    valid_mask1 = (
        (uv_int[:, 0] >= 0)
        & (uv_int[:, 0] < image_W)
        & (uv_int[:, 1] >= 0)
        & (uv_int[:, 1] < image_H)
    ).reshape(N, 8)

    valid_mask2 = ((depth > 0.5) & (depth < 40)).reshape(N, 8)
    gt_box2d_mask = valid_mask1.any(axis=1) & valid_mask2.all(axis=1)  # [N, ]

    gt_box2d = uv_int.reshape(N, 8, 2)  # [N, 8, 2]
    gt_box2d_u = np.clip(gt_box2d[:, :, 0], 0, image_W - 1)
    gt_box2d_v = np.clip(gt_box2d[:, :, 1], 0, image_H - 1)
    gt_box2d = np.stack((gt_box2d_u, gt_box2d_v), axis=-1)

    # create fg/bg mask
    fg_mask = np.zeros((image_H, image_W))

    if mask == "bin":
        fg_mask = draw_bin_mask(fg_mask, gt_box2d, gt_box2d_mask)
    else:
        fg_mask = draw_gaussian_mask(fg_mask, gt_box2d, gt_box2d_mask)

    # DEBUG = True
    # if DEBUG:
    #     from matplotlib import pyplot as plt
    #     plt.imshow(image)
    #     for i in range(N):
    #         if gt_box2d_mask[i]:
    #             coord2d = gt_box2d[i]
    #             for start, end in [(0, 1), (1, 2), (2, 3), (3, 0),
    #                            (0, 4), (1, 5), (2, 6), (3, 7),
    #                            (4, 5), (5, 6), (6, 7), (7, 4)]:
    #                 plt.plot(coord2d[[start,end]][:,0], coord2d[[start,end]][:,1], marker="o", c='g')
    #     plt.savefig(f"/dssg/home/acct-eezy/eezy-user1/yifanlu/OpenCOOD-main/vis_result/2d_gt_boxes/image_gt_box2d_{idx}.png", dpi=300)
    #     plt.clf()
    #     plt.imshow(fg_mask*255)
    #     plt.savefig(f"/dssg/home/acct-eezy/eezy-user1/yifanlu/OpenCOOD-main/vis_result/2d_gt_boxes/image_gt_box2d_{idx}_mask.png", dpi=300)
    #     plt.clf()

    return gt_box2d, gt_box2d_mask, fg_mask


def ue4_to_lss(camera_to_lidar_matrix):
    
    camera_to_lidar_matrix = np.linalg.inv(camera_to_lidar_matrix)
    
    # Create transformation matrix for coordinate change
    R = np.array([
    [0,  0,  1],    #  x_new =  y_old
    [1,  0, 0],    #  y_new = -z_old
    [0,  -1,  0]     #  z_new =  x_old
    ], dtype=np.float32)

    # 4 × 4 homogeneous version
    T = np.eye(4).astype(np.float32)
    T[:3, :3] = R
    transformed_matrix = np.matmul(camera_to_lidar_matrix, T)
    return transformed_matrix