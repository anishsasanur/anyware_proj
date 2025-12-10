import tkinter as tk
import json

class SquareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Square Placement")
        
        self.square_size = 40
        self.padding = 2
        self.num_squares = 1
        self.squares = []
        self.dragging = None
        self.offset_x = 0
        self.offset_y = 0
        
        self.setup_ui()
        self.create_squares()
        
    def setup_ui(self):
        control = tk.Frame(self.root)
        control.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(control, text="Squares:").pack(side=tk.LEFT)
        self.num_spin = tk.Spinbox(control, from_=1, to=6, width=5, command=self.update_squares)
        self.num_spin.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control, text="Save Positions", command=self.save_positions).pack(side=tk.LEFT, padx=10)
        
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg='white')
        self.canvas.pack(padx=5, pady=5)
        
        self.canvas_width = 800
        self.canvas_height = 600
        self.origin_x = self.canvas_width // 2
        self.origin_y = self.canvas_height - 50
        self.origin_y = (self.origin_y // self.square_size) * self.square_size
        
        self.draw_grid()
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
    def draw_grid(self):
        for x in range(0, self.canvas_width, self.square_size):
            self.canvas.create_line(x, 0, x, self.canvas_height, fill='lightgray')
        for y in range(0, self.canvas_height, self.square_size):
            self.canvas.create_line(0, y, self.canvas_width, y, fill='lightgray')
        
        self.canvas.create_line(0, self.origin_y, self.canvas_width, self.origin_y, 
                               fill='black', width=3)
        
        self.canvas.create_oval(self.origin_x-5, self.origin_y-5, 
                               self.origin_x+5, self.origin_y+5, fill='red')
        
    def create_squares(self):
        self.squares = []
        
        for i in range(self.num_squares):
            x = self.origin_x - self.square_size // 2
            y = self.origin_y - self.square_size * (i + 1)
            rect = self.canvas.create_rectangle(x + self.padding, y + self.padding, 
                                                x + self.square_size - self.padding, 
                                                y + self.square_size - self.padding, 
                                                fill='gray', outline='black', width=2)
            self.squares.append({'id': rect})
    
    def update_squares(self):
        self.num_squares = int(self.num_spin.get())
        self.canvas.delete('all')
        self.draw_grid()
        self.create_squares()
    
    def on_press(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        for sq in self.squares:
            if sq['id'] == item:
                self.dragging = sq
                coords = self.canvas.coords(item)
                self.offset_x = event.x - coords[0]
                self.offset_y = event.y - coords[1]
                break
                
    def on_drag(self, event):
        if self.dragging:
            new_x = event.x - self.offset_x
            new_y = event.y - self.offset_y
            
            if new_y + self.square_size > self.origin_y:
                new_y = self.origin_y - self.square_size
            
            if self.check_collision(new_x, new_y, self.dragging['id']):
                return
                
            self.canvas.coords(self.dragging['id'], new_x + self.padding, new_y + self.padding, 
                             new_x + self.square_size - self.padding, new_y + self.square_size - self.padding)
    
    def on_release(self, event):
        if self.dragging:
            self.apply_gravity()
        self.dragging = None
        
    def check_collision(self, x, y, current_id):
        for sq in self.squares:
            if sq['id'] == current_id:
                continue
            coords = self.canvas.coords(sq['id'])
            other_x = coords[0] - self.padding
            other_y = coords[1] - self.padding
            if not (x + self.square_size <= other_x or x >= other_x + self.square_size or
                   y + self.square_size <= other_y or y >= other_y + self.square_size):
                return True
        return False
        
    def apply_gravity(self):
        moved = True
        while moved:
            moved = False
            for sq in self.squares:
                coords = self.canvas.coords(sq['id'])
                x, y = coords[0] - self.padding, coords[1] - self.padding
                new_y = y + 1
                
                if new_y + self.square_size <= self.origin_y:
                    if not self.check_collision(x, new_y, sq['id']):
                        self.canvas.coords(sq['id'], x + self.padding, new_y + self.padding, 
                                         x + self.square_size - self.padding, new_y + self.square_size - self.padding)
                        moved = True
                        
    def save_positions(self):
        positions = []
        for i, sq in enumerate(self.squares):
            coords = self.canvas.coords(sq['id'])
            center_x = ((coords[0] + coords[2]) / 2 - self.origin_x) / self.square_size
            center_y = (self.origin_y - (coords[1] + coords[3]) / 2) / self.square_size
            positions.append({
                'square': i + 1,
                'x': center_x,
                'y': center_y
            })
        
        save_path = 'src/planning/planning/gui/block_plan.json'
        with open(save_path, 'w') as f:
            json.dump(positions, f, indent=2)
        print(f"Positions saved to {save_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SquareGUI(root)
    root.mainloop()