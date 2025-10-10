# /Users/djbray/Desktop/COMM/live_plotter.py
"""Live plotting for training metrics with epoch markers."""

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


class LivePlotter:
    """Real-time training metrics plotter (total loss + accuracy) with epoch markers."""

    def __init__(self, update_interval=1, max_points=10000, max_epoch_markers=50):
        """
        Args:
            update_interval: Redraw every N updates to reduce overhead.
            max_points: Maximum number of data points to keep in memory (older points are downsampled).
            max_epoch_markers: Maximum number of epoch markers to display (older ones are removed).
        """
        self.update_interval = update_interval
        self.max_points = max_points
        self.max_epoch_markers = max_epoch_markers

        # Counters & buffers
        self.batch_counter = 0
        self.loss_x, self.loss_y = [], []
        self.acc_x, self.acc_y = [], []

        # Keep handles for epoch markers so we can manage them if needed
        self.epoch_marker_lines = []   # list[(loss_ax_line, acc_ax_line)]
        self.epoch_marker_labels = []  # list[text objects]

        # Figure/axes
        plt.ion()
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.suptitle("Training Progress", fontsize=12)

        # Pre-create line objects used during update()
        (self.line_total_loss,) = self.ax_loss.plot([], [], label="Total Loss")
        (self.line_accuracy,) = self.ax_acc.plot([], [], label="Accuracy (%)")

        # Axes styling
        self.ax_loss.set_xlabel("Batch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend()

        self.ax_acc.set_xlabel("Batch")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_ylim(0, 100)
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.legend()

        self.fig.tight_layout()

    def update(self, total_loss: float, accuracy: float):
        """Append a point and update the figure."""
        self.batch_counter += 1

        self.loss_x.append(self.batch_counter)
        self.loss_y.append(float(total_loss))

        self.acc_x.append(self.batch_counter)
        self.acc_y.append(float(accuracy))

        # Downsample old data if we exceed max_points to save memory
        if len(self.loss_x) > self.max_points:
            self._downsample_data()

        # Update line data
        self.line_total_loss.set_data(self.loss_x, self.loss_y)
        self.line_accuracy.set_data(self.acc_x, self.acc_y)

        # Rescale views
        self.ax_loss.relim(); self.ax_loss.autoscale_view()
        self.ax_acc.relim();  self.ax_acc.autoscale_view()

        # Redraw occasionally
        if self.batch_counter % self.update_interval == 0:
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def _downsample_data(self):
        """Downsample data to keep memory usage under control.
        
        Keeps all recent points and downsamples older points by a factor of 2.
        """
        # Keep the most recent half at full resolution
        keep_recent = self.max_points // 2
        
        # Downsample the older half by taking every other point
        old_loss_x = self.loss_x[:-keep_recent:2]
        old_loss_y = self.loss_y[:-keep_recent:2]
        old_acc_x = self.acc_x[:-keep_recent:2]
        old_acc_y = self.acc_y[:-keep_recent:2]
        
        # Combine downsampled old data with recent data
        self.loss_x = old_loss_x + self.loss_x[-keep_recent:]
        self.loss_y = old_loss_y + self.loss_y[-keep_recent:]
        self.acc_x = old_acc_x + self.acc_x[-keep_recent:]
        self.acc_y = old_acc_y + self.acc_y[-keep_recent:]

    def add_epoch_marker(self, epoch_idx: int, x_pos: int | None = None):
        """
        Draw a vertical line at the end of an epoch on both subplots.

        Args:
            epoch_idx: Epoch number (1-based or 0-based; used only for the label).
            x_pos:     Batch index at which to place the marker. Defaults to the
                       current batch_counter (i.e., call this right after finishing an epoch).
        """
        if x_pos is None:
            x_pos = self.batch_counter

        # Draw lines on both axes. Use a light alpha so curves stay visible.
        line_loss = self.ax_loss.axvline(x=x_pos, linestyle="--", alpha=0.4)
        line_acc  = self.ax_acc.axvline(x=x_pos, linestyle="--", alpha=0.4)
        self.epoch_marker_lines.append((line_loss, line_acc))

        # Add small "E{epoch}" label above the top of each plot
        y_top_loss = self.ax_loss.get_ylim()[1]
        y_top_acc  = self.ax_acc.get_ylim()[1]
        txt1 = self.ax_loss.text(x_pos, y_top_loss, f"E{epoch_idx}",
                                 rotation=90, va="bottom", ha="center", fontsize=8, alpha=0.7)
        txt2 = self.ax_acc.text(x_pos, y_top_acc, f"E{epoch_idx}",
                                rotation=90, va="bottom", ha="center", fontsize=8, alpha=0.7)
        self.epoch_marker_labels.extend([txt1, txt2])

        # Remove old epoch markers if we have too many (to prevent memory bloat)
        while len(self.epoch_marker_lines) > self.max_epoch_markers:
            old_line_loss, old_line_acc = self.epoch_marker_lines.pop(0)
            try:
                old_line_loss.remove()
                old_line_acc.remove()
            except Exception:
                pass
            
            # Remove corresponding labels (2 per marker)
            if len(self.epoch_marker_labels) >= 2:
                for _ in range(2):
                    old_label = self.epoch_marker_labels.pop(0)
                    try:
                        old_label.remove()
                    except Exception:
                        pass

        # Make sure new artists render
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def clear_epoch_markers(self):
        """Optional helper to remove all existing epoch markers and labels."""
        for l1, l2 in self.epoch_marker_lines:
            try:
                l1.remove(); l2.remove()
            except Exception:
                pass
        for t in self.epoch_marker_labels:
            try:
                t.remove()
            except Exception:
                pass
        self.epoch_marker_lines.clear()
        self.epoch_marker_labels.clear()
        self.fig.canvas.draw_idle()

    def save(self, filepath="training_progress.png"):
        self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {filepath}")

    def close(self):
        try:
            plt.ioff()
        finally:
            plt.close(self.fig)
