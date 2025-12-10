import tkinter as tk
from tkinter import ttk


class BlockApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # ---- Window ----
        self.title("Stacking Blocks with Gravity")
        self.geometry("800x600")
        self.resizable(False, False)

        # ---- Physics ----
        self.gravity = 2.0
        self.canvas_width = 700
        self.canvas_height = 450
        self.block_size = 60

        self.velocities = {}
        self.blocks = []

        self.selected_block = None
        self.drag_start = (0, 0)
        self.block_colors = {}

        self._build_ui()
        self._create_blocks()
        self._physics_loop()

    # ---------- UI ----------
    def _build_ui(self):
        self.container = ttk.Frame(self, padding=10)
        self.container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.container,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white"
        )
        self.canvas.pack(pady=10)

        self.submit_btn = ttk.Button(self.container, text="Submit", command=self.submit)
        self.submit_btn.pack(pady=10)

        self.status = ttk.Label(self.container, text="")
        self.status.pack()

    # ---------- Blocks ----------
    def _create_blocks(self):
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]

        x, y = 80, 40

        for color in colors:
            block_id = self.canvas.create_rectangle(
                x, y,
                x + self.block_size,
                y + self.block_size,
                fill=color, outline="black", width=2
            )

            self.blocks.append(block_id)
            self.block_colors[block_id] = color
            self.velocities[block_id] = 0.0

            self.canvas.tag_bind(block_id, "<Button-1>", self.select_block)
            self.canvas.tag_bind(block_id, "<B1-Motion>", self.drag_block)
            self.canvas.tag_bind(block_id, "<ButtonRelease-1>", self.release_block)

            x += 90

    # ---------- Mouse ----------
    def select_block(self, event):
        self.selected_block = self.canvas.find_withtag("current")[0]
        self.velocities[self.selected_block] = 0.0
        self.drag_start = (event.x, event.y)
        self.canvas.tag_raise(self.selected_block)

    def drag_block(self, event):
        if not self.selected_block:
            return

        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]

        self.canvas.move(self.selected_block, dx, dy)
        self._apply_wall_constraints(self.selected_block)

        self.drag_start = (event.x, event.y)

    def release_block(self, _):
        self.selected_block = None

    # ---------- Physics ----------
    def _physics_loop(self):
        for block in self.blocks:
            if block == self.selected_block:
                continue

            self.velocities[block] += self.gravity
            self.canvas.move(block, 0, self.velocities[block])

            self._resolve_floor(block)
            self._resolve_block_collisions(block)
            self._apply_wall_constraints(block)

        self.after(16, self._physics_loop)

    # ---------- Collisions ----------
    def _resolve_floor(self, block):
        x1, y1, x2, y2 = self.canvas.coords(block)

        if y2 >= self.canvas_height:
            dy = self.canvas_height - y2
            self.canvas.move(block, 0, dy)
            self.velocities[block] = 0.0

    def _resolve_block_collisions(self, falling_block):
        fx1, fy1, fx2, fy2 = self.canvas.coords(falling_block)

        for other in self.blocks:
            if other == falling_block:
                continue

            ox1, oy1, ox2, oy2 = self.canvas.coords(other)

            horizontal_overlap = fx2 > ox1 and fx1 < ox2
            landed = (
                fy2 >= oy1 and
                fy1 < oy1 and
                abs(fy2 - oy1) <= abs(self.velocities[falling_block]) + 2
            )

            if horizontal_overlap and landed:
                dy = oy1 - fy2
                self.canvas.move(falling_block, 0, dy)
                self.velocities[falling_block] = 0.0
                return

    # ---------- Walls ----------
    def _apply_wall_constraints(self, block):
        x1, y1, x2, y2 = self.canvas.coords(block)

        dx = dy = 0

        if x1 < 0:
            dx = -x1
        elif x2 > self.canvas_width:
            dx = self.canvas_width - x2

        if y1 < 0:
            dy = -y1

        if dx or dy:
            self.canvas.move(block, dx, dy)
            self.velocities[block] = 0.0

    # ---------- Overlap Detection (For Submit Validation) ----------
    def _has_overlaps(self):
        for i in range(len(self.blocks)):
            x1, y1, x1b, y1b = self.canvas.coords(self.blocks[i])

            for j in range(i + 1, len(self.blocks)):
                x2, y2, x2b, y2b = self.canvas.coords(self.blocks[j])

                overlap = not (
                    x1b <= x2 or
                    x2b <= x1 or
                    y1b <= y2 or
                    y2b <= y1
                )

                if overlap:
                    return True
        return False

    # ---------- Submit ----------
    def submit(self):
        if self._has_overlaps():
            self.status.config(text="❌ Submission blocked: Cubes overlap!", foreground="red")
            return

        result = []

        for block in self.blocks:
            x1, y1, x2, y2 = self.canvas.coords(block)
            result.append({
                "color": self.block_colors[block],
                "x": round(x1),
                "y": round(y1),
                "size": self.block_size
            })

        print("\n✅ SUBMISSION ACCEPTED:")
        for i, r in enumerate(result, 1):
            print(f"Block {i}: {r}")

        self.status.config(text="✅ Submitted successfully!", foreground="green")

# ---------- Run ----------
def main():
    app = BlockApp()
    app.mainloop()


if __name__ == "__main__":
    main()
