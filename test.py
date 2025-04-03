import numpy as np
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import laspy

# Cloth grid creation
def create_cloth_grid(bbox, resolution):
    print("Creating cloth grid...")
    minx, miny, maxx, maxy = bbox
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_z = np.full_like(grid_x, fill_value=1000.0, dtype=np.float32)
    print(f"Cloth grid shape: {grid_z.shape}")
    return grid_x, grid_y, grid_z

# Cloth simulation step
def simulate_cloth(grid_x, grid_y, grid_z, inverted_points_tree, 
                   time_step=0.65, rigidness=1.0, iterations=50, gravity=0.2, threshold=0.5):
    print("Simulating cloth...")
    rows, cols = grid_z.shape
    for i in range(iterations):
        grid_z -= gravity
        positions = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=-1)
        distances, _ = inverted_points_tree.query(positions, k=1)
        collision_mask = distances < threshold
        grid_z.ravel()[collision_mask] += gravity

        z_new = grid_z.copy()
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                neighbors = [
                    grid_z[y-1, x], grid_z[y+1, x],
                    grid_z[y, x-1], grid_z[y, x+1]
                ]
                z_new[y, x] = (grid_z[y, x] + sum(neighbors)) / (len(neighbors) + 1)
        grid_z = (1 - rigidness) * grid_z + rigidness * z_new

        if i % 10 == 0 or i == iterations - 1:
            print(f"  Iteration {i+1}/{iterations} complete.")
    return grid_z

# Ground point classification
def classify_ground_points(original_points, cloth_surface, grid_x, grid_y, max_dist=0.5):
    print("Classifying ground points...")
    from scipy.interpolate import griddata
    cloth_z = griddata(
        (grid_x.ravel(), grid_y.ravel()),
        cloth_surface.ravel(),
        (original_points[:, 0], original_points[:, 1]),
        method='linear',
        fill_value=np.nan
    )
    dz = original_points[:, 2] - cloth_z
    ground_mask = np.abs(dz) < max_dist
    print(f"Ground points classified: {np.count_nonzero(ground_mask)} / {len(original_points)}")
    return ground_mask

# Main cloth filter pipeline
def cloth_filter(points, bbox, resolution=1.0, iterations=50, rigidness=0.3,
                 time_step=0.65, gravity=0.2, threshold=0.5, max_dist=0.5):
    print("Running cloth simulation filter...")
    inverted_points = points.copy()
    inverted_points[:, 2] *= -1
    kdtree = cKDTree(inverted_points)

    grid_x, grid_y, grid_z = create_cloth_grid(bbox, resolution)
    cloth_surface = simulate_cloth(
        grid_x, grid_y, grid_z, kdtree,
        time_step=time_step, rigidness=rigidness,
        iterations=iterations, gravity=gravity, threshold=threshold
    )
    ground_mask = classify_ground_points(points, cloth_surface, grid_x, grid_y, max_dist=max_dist)
    return ground_mask

# DEM export
def export_dem_to_tif(points, resolution, output_path, nodata_value=-9999):
    print(f"Exporting DEM to {output_path}...")
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    minx, miny = np.min(x), np.min(y)
    maxx, maxy = np.max(x), np.max(y)

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    dem = np.full((height, width), nodata_value, dtype=np.float32)
    counts = np.zeros_like(dem)

    ix = ((x - minx) / resolution).astype(int)
    iy = ((maxy - y) / resolution).astype(int)

    for i in range(len(points)):
        xi, yi = ix[i], iy[i]
        if 0 <= xi < width and 0 <= yi < height:
            if dem[yi, xi] == nodata_value:
                dem[yi, xi] = z[i]
            else:
                dem[yi, xi] += z[i]
            counts[yi, xi] += 1

    mask = counts > 0
    dem[mask] /= counts[mask]

    transform = from_origin(minx, maxy, resolution, resolution)

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(dem, 1)
    print("DEM export complete.")

# Point cloud visualization
def plot_point_cloud(points, color='terrain', title="Point Cloud", path=None):
    print(f"Plotting point cloud to {path}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap=color, s=0.2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(path)
    print("Point cloud plot saved.")

# === Main Script ===

print("Loading LAS file...")
las = laspy.read('/isipd/projects/p_planetdw/data/lidar/04_preprocessed/newtest/Tuk.las')
points = np.vstack((las.x, las.y, las.z)).T
print(f"Loaded {points.shape[0]} points.")

minx, miny, maxx, maxy = las.x.min(), las.y.min(), las.x.max(), las.y.max()
bbox = (minx, miny, maxx, maxy)

ground_mask = cloth_filter(points, bbox, resolution=1, iterations=50)
ground_points = points[ground_mask]

# Export DEM
export_dem_to_tif(
    ground_points,
    resolution=1,
    output_path='/isipd/projects/p_planetdw/data/lidar/04_preprocessed/newtest/Tuk.tif'
)

# Plot
plot_point_cloud(
    points,
    title="Original Point Cloud",
    path='/isipd/projects/p_planetdw/data/lidar/04_preprocessed/newtest/original_point_cloud.png'
)

print("All tasks complete.")
