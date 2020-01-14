import numpy as np
import trimesh
from trimesh.transformations import transform_points

colors = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
vertices = colors
scene = trimesh.scene.Scene()
for v, c in zip(vertices, colors):
    sphere = trimesh.primitives.Sphere(center=v, radius=0.1)
    # add point clouds - primitives don't render well
    vertex_colors = np.tile(
        np.expand_dims(c, axis=0)*255, (sphere.vertices.shape[0], 1))
    pc = trimesh.PointCloud(sphere.vertices, colors=vertex_colors)
    scene.add_geometry(pc)

scene.camera._resolution = (640,480)
print('# # camera resoltuion \n',scene.camera._resolution)
camera = scene.camera
transform = camera.transform
print("\n# # Camera extrinsic")
print(transform)
K = camera.K
print("\n# # Camera intrinsic")
print(K)
scene.show()

# render scene
from io import BytesIO
img = scene.save_image(resolution=(640,480))
from PIL import Image
rendered = Image.open(BytesIO(img)).convert("RGB")
rendered.save("rendered.jpg")

# numpy version

vertices = colors
transformed = transform_points(vertices, transform)  # transformation
projected = np.matmul(transformed, K.T)              # homogeneous projection
xy = projected[:, :2] / projected[:, 2:]             # make non-homogeneous

print("\n# # Projected 2D points\n",xy)


def show(xy):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,1)
    x, y = xy.T
    # ax = plt.gca()
    ax[0].scatter(x, y, c=colors)
    ax[1].imshow(rendered)
    plt.show()

show(xy)
