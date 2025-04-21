import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from scipy.stats import norm

# === PDF Calculation ===
def normal_pdf(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

def pdf_2d_quadrature_bm(p: float, alpha: float, num_pts=1000):
    dt = 0.5
    thetas = np.linspace(0, 2 * np.pi, num_pts)
    dx1 = p * np.cos(thetas)
    dx2 = p * np.sin(thetas)

    x1 = dx1 * np.sqrt(dt)
    x2 = (dx1 + dx2) * np.sqrt(dt)

    phi_vals = normal_pdf(dx1, 0, 1) * normal_pdf(dx2, 0, 1)
    weights = phi_vals * p * (2 * np.pi / num_pts)  # arc length elements

    if alpha == 0.:
        total_weight = 1.
    elif alpha == 0.5:
        total_weight = 0.7458913437205545
    elif alpha == 1.:
        total_weight = 0.37064413336206625
    elif alpha == 1.5:
        total_weight = 0.14605801048951172
    elif alpha == 2.0:
        total_weight = 0.047295252164004084
    else:
        raise NotImplementedError

    exit_weight = np.sum(weights[(np.abs(x1) > (alpha - 1e-5)) | (np.abs(x2) > (alpha - 1e-5))])

    if exit_weight == total_weight == 0.:
        result = 0.
    else:
        result = exit_weight / total_weight
    return result

# === Animation Generator ===
def generate_growing_circle_with_pdf(alpha):
    t = np.sqrt(0.5)
    max_radius = np.sqrt(5) * alpha / t
    x_lim = y_lim = 1.2 * max_radius
    x = np.linspace(-x_lim, x_lim, 500)

    frame_dir = "frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame_data = []
    pdf_data = []

    def segment_line(x, y, mask):
        x_seg = np.copy(x)
        y_seg = np.copy(y)
        x_seg[~mask] = np.nan
        y_seg[~mask] = np.nan
        return x_seg, y_seg

    def plot_frame(radius, frame_idx, pdf_data, duration=0.05):
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax = axs[0]
        ax2 = axs[1]

        # === Top Plot: Circle + Lines ===
        ax.plot(x, alpha/t - x, label=r'$\alpha/t - x$', color='blue')
        ax.plot(x, -alpha/t - x, label=r'$-\alpha/t - x$', color='blue')
        ax.axvline(alpha/t, color='blue', label=r'$x = \alpha/t$')
        ax.axvline(-alpha/t, color='blue', label=r'$x = -\alpha/t$')

        # Circle
        theta = np.linspace(0, 2*np.pi, 1000)
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)

        # Region masks
        cond_green = (cy > (alpha / t - cx)) | (cy < (-alpha / t - cx))
        cond_yellow_right = (cx > (alpha / t)) & (cy <= (alpha / t - cx))
        cond_yellow_left = (cx < (-alpha / t)) & (cy >= (-alpha / t - cx))
        cond_yellow = cond_yellow_right | cond_yellow_left
        cond_red = ~(cond_green | cond_yellow)

        rx, ry = segment_line(cx, cy, cond_red)
        gx, gy = segment_line(cx, cy, cond_green)
        yx, yy = segment_line(cx, cy, cond_yellow)

        ax.plot(rx, ry, color='red', linewidth=3)
        ax.plot(gx, gy, color='darkgreen', linewidth=3)
        ax.plot(yx, yy, color='lightgreen', linewidth=3)

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect('equal')
        ax.set_title(f"Radius = {radius:.2f}")
        ax.legend(loc='upper right')

        # === Bottom Plot: PDF vs Radius ===
        pdf_val = pdf_2d_quadrature_bm(radius, alpha)
        pdf_data.append((radius, pdf_val))

        r_vals, pdf_vals = zip(*pdf_data)
        ax2.plot(r_vals, pdf_vals, color='purple', linewidth=2)
        ax2.set_xlim(0, max_radius)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Radius")
        ax2.set_ylabel("Escape Probability")
        ax2.set_title("Probability Density vs Radius")

        filename = os.path.join(frame_dir, f"frame_{frame_idx:03d}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        frame_data.append((filename, duration))

    # === Animate ===
    r1 = alpha
    r2 = alpha / t
    r3 = max_radius
    num_steps = 30
    num_steps2 = 50

    for r in np.linspace(0, r1, num_steps):
        plot_frame(r, len(frame_data), pdf_data)

    for r in np.linspace(r1, r2, num_steps):
        plot_frame(r, len(frame_data), pdf_data)

    for r in np.linspace(r2, r3, num_steps2):
        plot_frame(r, len(frame_data), pdf_data)

    # === Save GIF ===
    with imageio.get_writer("growing_circle_with_pdf.gif", mode='I') as writer:
        for filename, duration in frame_data:
            image = imageio.imread(filename)
            writer.append_data(image, {"duration": duration})

    # === Clean Up ===
    for fname, _ in frame_data:
        os.remove(fname)
    os.rmdir(frame_dir)

# Run the generator
generate_growing_circle_with_pdf(alpha=1.0)

