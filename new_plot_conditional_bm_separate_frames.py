import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
    min_radius = alpha
    mid_radius = alpha / t
    max_radius = np.sqrt(10) * alpha
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

    def plot_circle_of_radius(radius, ax, outside_color, first_line, second_line):
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

        ax.plot(rx, ry, color=outside_color, linewidth=3)
        ax.plot(gx, gy, color=first_line, linewidth=3)
        ax.plot(yx, yy, color=second_line, linewidth=3)

    def plot_frame(radius, frame_idx, pdf_data, duration=0.05):
        # =========================
        # === TOP FIGURE (circle)
        # =========================
        fig1, ax = plt.subplots(figsize=(6, 6))

        ax.plot(x, alpha/t - x, label=r'$\alpha/\sqrt{\Delta t} - x$', color='blue')
        ax.plot(x, -alpha/t - x, label=r'$-\alpha/\sqrt{\Delta t} - x$', color='blue')
        ax.axvline(alpha/t, color='blue', label=r'$x = \alpha/\sqrt{\Delta t}$')
        ax.axvline(-alpha/t, color='blue', label=r'$x = -\alpha/\sqrt{\Delta t}$')

        plot_circle_of_radius(min_radius, ax, 'red', 'red', 'red')
        plot_circle_of_radius(mid_radius, ax, 'darkgreen', 'darkgreen', 'darkgreen')
        plot_circle_of_radius(max_radius, ax, 'lightgreen', 'lightgreen', 'lightgreen')

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_xlabel(r"$\Delta X_1/\sqrt{\Delta t}$")
        ax.set_ylabel(r"$\Delta X_2/\sqrt{\Delta t}$")

        # Parallelogram
        a = alpha / t
        corner_height = max_radius - 0.15
        parallelogram_vertices = np.array([
            [-a,  corner_height],
            [ a,  0.0],
            [ a,  -corner_height],
            [-a,  0.0],
        ])

        par_patch = Polygon(
            parallelogram_vertices,
            closed=True,
            facecolor='lightblue',
            edgecolor='none',
            alpha=0.3,
            zorder=0
        )
        ax.add_patch(par_patch)

        filename1 = os.path.join(frame_dir, f"frame_{frame_idx:03d}_top.pdf")
        plt.tight_layout()
        plt.savefig(filename1)
        plt.close(fig1)

        # =========================
        # === BOTTOM FIGURE (PDF)
        # =========================
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        pdf_val = pdf_2d_quadrature_bm(radius, alpha)
        pdf_data.append((radius, pdf_val))

        r_vals, pdf_vals = zip(*pdf_data)
        ax2.plot(r_vals, pdf_vals, color='purple', linewidth=2)

        ax2.set_xlim(0, max_radius + 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Radius")
        ax2.set_ylabel("Exit Probability")
        ax2.set_title("Probability Density vs Radius")

        ax2.axvline(x=min_radius, linestyle='--', color='red')
        ax2.axvline(x=mid_radius, linestyle='--', color='darkgreen')
        ax2.axvline(x=max_radius, linestyle='--', color='lightgreen')

        filename2 = os.path.join(frame_dir, f"frame_{frame_idx:03d}_bottom.pdf")
        plt.tight_layout()
        plt.savefig(filename2)
        plt.close(fig2)

        # Store only one set if you're still making GIFs (PNG recommended instead)
        frame_data.append((filename1, duration))

    # === Animate ===
    r1 = alpha
    r2 = alpha / t
    r3 = max_radius + 0.5
    num_steps = 30
    num_steps2 = 50

    for r in np.linspace(0, r1, num_steps):
        plot_frame(r, len(frame_data), pdf_data)

    for r in np.linspace(r1, r2, num_steps):
        plot_frame(r, len(frame_data), pdf_data)

    for r in np.linspace(r2, r3, num_steps2):
        plot_frame(r, len(frame_data), pdf_data)

    # === Save GIF ===
    # in order to make the gif, the frames must be pngs, not pdfs
    with imageio.get_writer("growing_circle_with_pdf.gif", mode='I') as writer:
        for filename, duration in frame_data:
            image = imageio.imread(filename)
            writer.append_data(image, {"duration": duration})

    # # # === Clean Up ===
    # for fname, _ in frame_data:
    #     os.remove(fname)
    # os.rmdir(frame_dir)

# Run the generator
generate_growing_circle_with_pdf(alpha=0.5)

