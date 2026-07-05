#!/usr/bin/env python3
"""
OpenPI Debug Data Visualizer
Displays frame images and corresponding action data with comparison between inference and replay
"""

import glob
import os
import re
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


class OpenPIDebugVisualizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.current_timestep = 0

        # Get all timesteps
        action_files = glob.glob(os.path.join(data_dir, "action_*.npy"))
        # Filter files to get unique timesteps (action_XXXX.npy or action_XXXX_inference.npy)
        timesteps_set = set()
        for f in action_files:
            # Match action_XXXX.npy or action_XXXX_inference.npy or action_XXXX_replay.npy
            match = re.search(r'action_(\d+)(?:_inference|_replay)?\.npy', f)
            if match:
                timesteps_set.add(int(match.group(1)))
        self.timesteps = sorted(list(timesteps_set))

        print(f"Found {len(self.timesteps)} timesteps: {self.timesteps}")

        # Create main window
        self.root = tk.Tk()
        self.root.title("OpenPI Debug Data Visualizer")
        self.root.geometry("1600x950")
        self.root.configure(bg='#f0f0f0')

        # Set style
        self.setup_style()
        self.setup_ui()
        self.load_timestep(0)

    def setup_style(self):
        """Setup UI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'), foreground='#34495e')
        style.configure('TButton', font=('Segoe UI', 9), padding=6)
        style.configure('Action.TButton', font=('Segoe UI', 9, 'bold'))
        
        # LabelFrame styling
        style.configure('TLabelframe', background='#f0f0f0', borderwidth=2, relief='groove')
        style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'), foreground='#2c3e50')

    def setup_ui(self):
        """Setup user interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🎯 OpenPI Debug Data Visualizer", 
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 15))

        # Control panel
        control_frame = self.create_control_panel(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))

        # Content area (images + actions)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left side: Camera views
        left_frame = self.create_camera_panel(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right side: Action data
        right_frame = self.create_action_panel(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Playback control variables
        self.playing = False
        self.play_job = None

    def create_control_panel(self, parent):
        """Create control panel with navigation and playback controls"""
        panel = ttk.LabelFrame(parent, text="⚙️ Controls", padding=15)
        
        # Timestep control section
        timestep_frame = ttk.Frame(panel)
        timestep_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        ttk.Label(timestep_frame, text="Timestep:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
        
        self.timestep_var = tk.StringVar()
        self.timestep_combo = ttk.Combobox(
            timestep_frame,
            textvariable=self.timestep_var,
            values=[f"{ts:04d}" for ts in self.timesteps],
            state="readonly",
            width=12,
            font=('Consolas', 10)
        )
        self.timestep_combo.pack(side=tk.LEFT, padx=5)
        self.timestep_combo.bind("<<ComboboxSelected>>", self.on_timestep_change)
        
        # Info label
        self.info_label = ttk.Label(
            timestep_frame, 
            text=f"/ {len(self.timesteps)} total",
            foreground='#7f8c8d'
        )
        self.info_label.pack(side=tk.LEFT, padx=(5, 0))

        # Navigation buttons
        nav_frame = ttk.Frame(panel)
        nav_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        ttk.Label(nav_frame, text="Navigate:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(nav_frame, text="◀ Previous", command=self.prev_timestep).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_timestep).pack(side=tk.LEFT, padx=2)

        # Playback controls
        play_frame = ttk.Frame(panel)
        play_frame.pack(side=tk.LEFT)
        
        ttk.Label(play_frame, text="Playback:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(play_frame, text="▶ Play", command=self.play_sequence, style='Action.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(play_frame, text="⏸ Stop", command=self.stop_play).pack(side=tk.LEFT, padx=2)
        
        return panel

    def create_camera_panel(self, parent):
        """Create camera views panel"""
        panel = ttk.LabelFrame(parent, text="📷 Camera Views", padding=15)
        
        # Agentview camera
        agent_frame = ttk.Frame(panel)
        agent_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        ttk.Label(agent_frame, text="Agentview Camera", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        agent_container = tk.Frame(agent_frame, bg='#2c3e50', relief=tk.SUNKEN, borderwidth=2)
        agent_container.pack()
        self.agent_image_label = ttk.Label(agent_container)
        self.agent_image_label.pack(padx=2, pady=2)

        # Eye-in-hand camera
        eye_frame = ttk.Frame(panel)
        eye_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(eye_frame, text="Eye-in-Hand Camera", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        eye_container = tk.Frame(eye_frame, bg='#2c3e50', relief=tk.SUNKEN, borderwidth=2)
        eye_container.pack()
        self.eye_image_label = ttk.Label(eye_container)
        self.eye_image_label.pack(padx=2, pady=2)
        
        return panel

    def create_action_panel(self, parent):
        """Create action data panel"""
        panel = ttk.LabelFrame(parent, text="🎮 Action Data", padding=15)
        
        # Action sequence table
        self.create_action_table(panel)
        
        # Action visualization plot
        self.create_action_plot(panel)
        
        return panel

    def create_action_table(self, parent):
        """Create action data table"""
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(
            table_frame, 
            text="Action Sequence (10 steps):", 
            style='Header.TLabel'
        ).pack(anchor=tk.W, pady=(0, 8))

        # Create table
        columns = ["Step"] + [f"Dim {i}" for i in range(8)]
        self.action_tree = ttk.Treeview(
            table_frame, 
            columns=columns, 
            show="headings", 
            height=8
        )

        # Configure columns
        self.action_tree.column("Step", width=60, anchor=tk.CENTER)
        self.action_tree.heading("Step", text="Step")
        for i in range(8):
            col = f"Dim {i}"
            self.action_tree.column(col, width=80, anchor=tk.CENTER)
            self.action_tree.heading(col, text=col)

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.action_tree.yview)
        self.action_tree.configure(yscrollcommand=scrollbar.set)

        self.action_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Alternate row colors
        self.action_tree.tag_configure('oddrow', background='#f9f9f9')
        self.action_tree.tag_configure('evenrow', background='#ffffff')

    def create_action_plot(self, parent):
        """Create action data plot"""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            plot_frame, 
            text="Action Visualization (Inference vs Replay):", 
            style='Header.TLabel'
        ).pack(anchor=tk.W, pady=(0, 8))

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_timestep(self, index):
        """Load data for specified timestep"""
        if 0 <= index < len(self.timesteps):
            self.current_timestep = index
            timestep = self.timesteps[index]

            # Update combobox
            self.timestep_var.set(f"{timestep:04d}")

            # Load images
            self.load_images(timestep)

            # Load action data
            self.load_actions(timestep)

    def load_images(self, timestep):
        """Load camera images"""
        try:
            # Agentview camera
            agent_path = os.path.join(self.data_dir, f"frame_{timestep:04d}_agentview.png")
            if os.path.exists(agent_path):
                agent_img = Image.open(agent_path)
                agent_img = agent_img.resize((400, 300), Image.Resampling.LANCZOS)
                agent_photo = ImageTk.PhotoImage(agent_img)
                self.agent_image_label.configure(image=agent_photo)
                self.agent_image_label.image = agent_photo  # Keep reference

            # Eye-in-hand camera
            eye_path = os.path.join(self.data_dir, f"frame_{timestep:04d}_eye.png")
            if os.path.exists(eye_path):
                eye_img = Image.open(eye_path)
                eye_img = eye_img.resize((400, 300), Image.Resampling.LANCZOS)
                eye_photo = ImageTk.PhotoImage(eye_img)
                self.eye_image_label.configure(image=eye_photo)
                self.eye_image_label.image = eye_photo  # Keep reference

        except Exception as e:
            print(f"Error loading images: {e}")

    def load_actions(self, timestep):
        """Load action data"""
        try:
            # Try multiple file name formats
            action_paths = [
                os.path.join(self.data_dir, f"action_{timestep:04d}.npy"),
                os.path.join(self.data_dir, f"action_{timestep:04d}_inference.npy"),
            ]
            
            action_path = None
            for path in action_paths:
                if os.path.exists(path):
                    action_path = path
                    break
            
            if action_path:
                actions = np.load(action_path)

                # Update table
                for item in self.action_tree.get_children():
                    self.action_tree.delete(item)

                for i, action in enumerate(actions):
                    values = [f"Step {i}"] + [f"{val:.4f}" for val in action]
                    tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                    self.action_tree.insert("", tk.END, values=values, tags=(tag,))

                # Update plot
                self.update_action_plot(actions, timestep)
            else:
                print(f"No action file found for timestep {timestep}")

        except Exception as e:
            print(f"Error loading action data: {e}")

    def update_action_plot(self, actions, timestep):
        """Update action plot with inference and replay comparison"""
        self.ax.clear()

        # Set style
        self.ax.set_facecolor('#ffffff')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        # Color palette for dimensions
        colors_inference = plt.cm.tab10(np.linspace(0, 1, actions.shape[1]))
        colors_replay = plt.cm.Set3(np.linspace(0, 1, actions.shape[1]))

        # Plot inference actions (solid lines)
        for dim in range(actions.shape[1]):
            self.ax.plot(
                actions[:, dim], 
                label=f"Inference Dim {dim}", 
                marker='o', 
                markersize=5,
                linewidth=2,
                color=colors_inference[dim],
                alpha=0.8
            )

        # Try to load replay data for comparison
        replay_path = os.path.join(self.data_dir, f"action_{timestep:04d}_replay.npy")
        if os.path.exists(replay_path):
            replay_actions = np.load(replay_path)
            for dim in range(min(replay_actions.shape[1], actions.shape[1])):
                self.ax.plot(
                    replay_actions[:, dim], 
                    label=f"Replay Dim {dim}", 
                    marker='x', 
                    markersize=6,
                    linewidth=1.5,
                    linestyle='--',
                    color=colors_replay[dim],
                    alpha=0.6
                )

        self.ax.set_xlabel("Action Step", fontsize=11, fontweight='bold')
        self.ax.set_ylabel("Action Value", fontsize=11, fontweight='bold')
        self.ax.set_title(
            f"Timestep {self.timesteps[self.current_timestep]:04d} - Action Sequence", 
            fontsize=12, 
            fontweight='bold',
            pad=15
        )
        self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, framealpha=0.9)
        self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        plt.tight_layout()
        self.canvas.draw()

    def on_timestep_change(self, event=None):
        """Handle timestep change event"""
        selected = self.timestep_var.get()
        if selected:
            timestep = int(selected)
            index = self.timesteps.index(timestep)
            self.load_timestep(index)

    def prev_timestep(self):
        """Go to previous timestep"""
        if self.current_timestep > 0:
            self.load_timestep(self.current_timestep - 1)

    def next_timestep(self):
        """Go to next timestep"""
        if self.current_timestep < len(self.timesteps) - 1:
            self.load_timestep(self.current_timestep + 1)

    def play_sequence(self):
        """Play sequence animation"""
        if not self.playing:
            self.playing = True
            self.play_next()

    def play_next(self):
        """Play next frame"""
        if self.playing:
            self.next_timestep()
            if self.current_timestep < len(self.timesteps) - 1:
                self.play_job = self.root.after(500, self.play_next)  # 500ms interval
            else:
                self.playing = False

    def stop_play(self):
        """Stop playback"""
        self.playing = False
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    data_dir = os.environ.get(
        "OPENPI_DEBUG_DIR",
        "benchmarks/openpi/openpi_libero_debug",
    )

    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        return

    print("Starting OpenPI Debug Data Visualizer...")
    visualizer = OpenPIDebugVisualizer(data_dir)
    visualizer.run()


if __name__ == "__main__":
    main()
