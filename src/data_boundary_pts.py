import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


def sample_circle(num_landmarks: int, radius=1.0, centre=jnp.asarray([0, 0])) -> jnp.ndarray:
    theta = jnp.linspace(0, 2 * jnp.pi, num_landmarks, endpoint=False)
    x = jnp.cos(theta)
    y = jnp.sin(theta)
    return (radius * jnp.stack([x, y], axis=1) + centre).flatten()


def y_uniform_circle(key, num_samples, r=1.0):
    theta = jax.random.uniform(key, (num_samples, 1), minval=0, maxval=2 * jax.numpy.pi)
    return jnp.concatenate([r * jax.numpy.cos(theta), r * jax.numpy.sin(theta)], axis=-1)


def plot_functions(functions):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each function
    for i, f in enumerate(functions):
        x, y = f()
        ax.plot(x, y)

    # Add grid, legend, and title
    ax.grid(True)
    ax.legend()
    plt.show()


def smiley_face_fns(num_eye, num_brow, num_mouth, num_outline):
    def left_eye():
        x = np.linspace(-2.4, -0.6, num_eye)
        y = -((x + 1.4) ** 2 / 5 - 0.2)
        return jnp.stack([x, y], axis=1).flatten()

    def right_eye():
        x = np.linspace(0.6, 2.4, num_eye)
        y = -((x - 1.4) ** 2 / 5 - 0.2)
        return jnp.stack([x, y], axis=1).flatten()

    def left_brow():
        x = np.linspace(-3, -1, num_brow)
        y = -(((x + 1.5) / 2.6) ** 2) + 1.8
        return jnp.stack([x, y], axis=1).flatten()

    def right_brow():
        x = np.linspace(1, 3, num_brow)
        y = -(((x - 1.5) / 2.6) ** 2) + 1.8
        return jnp.stack([x, y], axis=1).flatten()

    def mouth():
        x = np.linspace(-1.3, 1.3, num_mouth)
        y = (x**2) / 2.5 - 2.7
        return jnp.stack([x, y], axis=1).flatten()

    def outline():
        theta = np.linspace(0, 2 * np.pi, num_outline, endpoint=True)
        x = np.cos(theta)
        y = np.sin(theta)
        r = 4.5
        return jnp.stack([r * x, r * y], axis=1).flatten()

    return [left_eye, right_eye, left_brow, right_brow, mouth, outline]


def pensive_face_fns(num_eye, num_brow, num_mouth, num_outline):
    def left_eye():
        x = np.linspace(-3, -0.6, num_eye)
        y = (x + 1.8) ** 2 / 2.5 - 1.2
        return jnp.stack([x, y], axis=1).flatten()

    def right_eye():
        x = np.linspace(0.6, 3, num_eye)
        y = (x - 1.8) ** 2 / 2.5 - 1.2
        return jnp.stack([x, y], axis=1).flatten()

    def left_brow():
        x = np.linspace(-3.5, -1.6, num_brow)
        y = (x + 3.5) ** 2 / 4 + 0.6
        return jnp.stack([x, y], axis=1).flatten()

    def right_brow():
        x = np.linspace(1.6, 3.5, num_brow)
        y = (x - 3.5) ** 2 / 4 + 0.6
        return jnp.stack([x, y], axis=1).flatten()

    def mouth():
        x = np.linspace(-1, 1, num_mouth)
        y = -2.8 * np.ones_like(x)
        return jnp.stack([x, y], axis=1).flatten()

    def outline():
        theta = np.linspace(0, 2 * np.pi, num_outline, endpoint=True)
        x = np.cos(theta)
        y = np.sin(theta)
        r = 4.5
        return jnp.stack([r * x, r * y], axis=1).flatten()

    return [left_eye, right_eye, left_brow, right_brow, mouth, outline]


def flattened_array_from_faces(fns):
    pts = tuple(f() for f in fns)
    xy = jnp.concatenate(pts, axis=-1).T.flatten()
    xy = xy / jnp.max(jnp.abs(xy))
    return xy


def butterfly(path="../data/inverted_butterfly_tom.png", remove_pts=-2, num_landmarks=100):
    def order_points(unordered_points):
        points_for_waste = unordered_points.copy()
        p0 = points_for_waste[0, :].copy()
        points = [p0]
        for _ in range(len(unordered_points)):
            Midx = spatial.distance_matrix(p0[None, :], points_for_waste)
            Midx[Midx == 0.0] = 100
            idx = np.argmin(Midx)
            p0 = points_for_waste[idx, :].copy()
            points_for_waste[idx, :] = 100000
            points.append(p0)
        return np.asarray(points)

    def get_points(im_path: str):
        image = cv2.imread(im_path)
        gray_scale_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges_in_im = cv2.Canny(gray_scale_im, 50, 200)
        pixels = np.argwhere(edges_in_im == 255)
        points = np.array([1, -1]) * pixels[:, ::-1]  # Rotate points
        return points

    def _interpolate():
        pts = get_points(path)
        ordered = np.asarray(order_points(pts), dtype=float)[:remove_pts]
        ordered[:, 0] = _scale(ordered[:, 0])
        ordered[:, 1] = _scale(ordered[:, 1])
        x1 = np.interp(np.arange(0.0, len(ordered), 0.05), np.arange(len(ordered)), ordered[:, 0])
        x2 = np.interp(np.arange(0.0, len(ordered), 0.05), np.arange(len(ordered)), ordered[:, 1])
        return x1, x2

    def _scale(points):
        maxpt = np.max(points)
        minpt = np.min(points)
        return (points - minpt) / (maxpt - minpt)

    x, y = _interpolate()
    step = len(x) // (num_landmarks - 1)
    indices = jnp.arange(0, len(x), step)
    x = x[indices]
    y = y[indices]
    return jnp.stack([x, y], axis=1).flatten()


def butterfly_tom(num_landmarks=100):
    return butterfly("../../data/inverted_butterfly_tom.png", -2, num_landmarks)


def butterfly1(num_landmarks=100):
    path = "../../data/inverted_butterfly1.png"
    remove_pts = -466
    return butterfly(path, remove_pts, num_landmarks)


def butterfly2(num_landmarks=100):
    path = "../../data/inverted_butterfly2.png"
    remove_pts = -70
    return butterfly(path, remove_pts, num_landmarks)
