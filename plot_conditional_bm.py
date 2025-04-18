import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


# Helper function to break at discontinuities using NaN
def segment_line(x, y, mask):
    x_seg = np.copy(x)
    y_seg = np.copy(y)
    x_seg[~mask] = np.nan
    y_seg[~mask] = np.nan
    return x_seg, y_seg

def generate_growing_circle_gif(alpha):
    t = np.sqrt(1/2)
    max_radius = np.sqrt(5) * alpha / t
    margin = 1.1
    x_lim = y_lim = margin * max_radius
    x = np.linspace(-x_lim, x_lim, 500)
    
    # Directory for frames
    frame_dir = "frames"
    os.makedirs(frame_dir, exist_ok=True)
    filenames = []
    frame_data = []

    def plot_frame(radius, frame_idx, duration=0.5):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the four lines
        ax.plot(x, alpha/t - x, label=r'$\alpha/t - x$', color='blue')
        ax.plot(x, -alpha/t - x, label=r'$-\alpha/t - x$', color='blue')
        ax.axvline(alpha/t, color='blue', label=r'$x = \alpha/t$')
        ax.axvline(-alpha/t, color='blue', label=r'$x = -\alpha/t$')

        # Circle coordinates
        theta = np.linspace(0, 2*np.pi, 1000)
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)

        # Conditions for green highlight
        cond_above_line = cy > (alpha / t - cx)
        cond_below_line = cy < (-alpha / t - cx)
        cond_green = cond_above_line | cond_below_line

        # Yellow region: combine left and right yellow regions
        cond_yellow_right = (cx > (alpha / t)) & (cy <= (alpha / t - cx))
        cond_yellow_left = (cx < (-alpha / t)) & (cy >= (-alpha / t - cx))
        cond_yellow = cond_yellow_right | cond_yellow_left

        # Red is the remainder (not green and not yellow)
        cond_red = ~(cond_green | cond_yellow)

        # Create red and green segments as single arrays with NaN-separated gaps
        gx, gy = segment_line(cx, cy, cond_green)
        rx, ry = segment_line(cx, cy, cond_red)
        yx, yy = segment_line(cx, cy, cond_yellow)
        
        ax.plot(rx, ry, color='red', linewidth=4.5)
        ax.plot(gx, gy, color='darkgreen', linewidth=4.5)
        ax.plot(yx, yy, color='lightgreen', linewidth=4.5)

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_title(f"Radius = {radius:.2f}")

        filename = os.path.join(frame_dir, f"frame_{frame_idx:03d}.png")
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)

        frame_data.append((filename, duration))

    # Radius checkpoints
    r1 = alpha
    r2 = alpha / t
    r3 = np.sqrt(5) * alpha / t
    num_steps1 = 30
    num_steps2 = int(num_steps1 * r2 / r1)
    num_steps3 = int(num_steps1 * r3 / r2)

    # Stage 1: Expand to r1
    for r in np.linspace(0, r1, num_steps1):
        plot_frame(r, len(frame_data))

    # Stage 2: Expand to r2
    for r in np.linspace(r1, r2, num_steps1):
        plot_frame(r, len(frame_data))

    # Stage 3: Expand to r3
    for r in np.linspace(r2, r3, num_steps3):
        plot_frame(r, len(frame_data))

    # Create gif
    with imageio.get_writer("growing_circle.gif", mode='I') as writer:
        for filename, duration in frame_data:
            image = imageio.imread(filename)
            writer.append_data(image, {"duration": duration})

    # Cleanup
    for fname in filenames:
        os.remove(fname)
    os.rmdir(frame_dir)

# Run the animation generator
generate_growing_circle_gif(alpha=1.0)


