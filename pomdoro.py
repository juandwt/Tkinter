import tkinter as tk
from datetime import datetime, timedelta

colors = {"dark_green": "#4a6c65", "olive": "#7B904B"}

class PomodoroTimer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pomodoro Timer")
        
        self.work_time = 25 * 60  # 25 minutos
        self.break_time = 5 * 60  # 5 minutos
        self.current_time = self.work_time
        self.running = False
        self.on_break = False
        self.completed_pomodoros = 0

        self.left_half = tk.Frame(self.root, bg=colors["dark_green"])
        self.left_half.pack(side=tk.LEFT, fill="both", expand=1)

        self.right_half = tk.Frame(self.root, bg="#FFFFFF")
        self.right_half.pack(side=tk.RIGHT, fill="both", expand=1)
        
        #self.label = tk.Label(self.left_half, text="Pomodoro Timer", font=("Helvetica", 18), bg=colors["dark_green"], fg="white")
        #self.label.place(relx=0.5, rely=0.2, anchor="center")

        self.timer_label = tk.Label(self.right_half, text=self.format_time(self.current_time), font=("Helvetica", 48), bg="#FFFFFF", fg="black")
        self.timer_label.place(relx=0.5, rely=0.5, anchor="center")

        self.start_button = tk.Button(self.left_half, text="Start", command=self.start_timer, bg="#4a6c65", fg="white", borderwidth=0, highlightthickness=0, relief="flat")
        self.start_button.place(relx=0.25, rely=0.5, anchor="center")

        self.stop_button = tk.Button(self.left_half, text="Stop", command=self.stop_timer, bg="#4a6c65", fg="white", borderwidth=0, highlightthickness=0, relief="flat")
        self.stop_button.place(relx=0.5, rely=0.5, anchor="center")

        self.reset_button = tk.Button(self.left_half, text="Reset", command=self.reset_timer, bg="#4a6c65", fg="white", borderwidth=0, highlightthickness=0, relief="flat")
        self.reset_button.place(relx=0.75, rely=0.5, anchor="center")

        # Agregar los 4 círculos en la parte inferior
        self.circle_frames = []
        for i in range(4):
            circle_frame = tk.Frame(self.right_half, width=10, height=10, bg="white", borderwidth=1, relief="solid")
            circle_frame.place(relx=0.35 + i * 0.1, rely=0.8, anchor="center")
            self.circle_frames.append(circle_frame)

    def format_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02}:{seconds:02}"

    def start_timer(self):
        if not self.running:
            self.running = True
            self.update_timer()

    def stop_timer(self):
        self.running = False

    def reset_timer(self):
        self.running = False
        self.current_time = self.work_time if not self.on_break else self.break_time
        self.timer_label.config(text=self.format_time(self.current_time))

    def update_timer(self):
        if self.running:
            if self.current_time > 0:
                self.current_time -= 1
                self.timer_label.config(text=self.format_time(self.current_time))
                self.root.after(1000, self.update_timer)
            else:
                self.running = False
                self.on_break = not self.on_break
                self.current_time = self.break_time if self.on_break else self.work_time
                self.timer_label.config(text=self.format_time(self.current_time))
                self.start_timer()
                if not self.on_break:
                    self.complete_pomodoro()

    def complete_pomodoro(self):
        if self.completed_pomodoros < 4:
            self.completed_pomodoros += 1
            for i in range(self.completed_pomodoros):
                self.circle_frames[i].configure(bg=colors["dark_green"])

if __name__ == "__main__":
    root = tk.Tk()
    root.config(bg="#FFFFFF")
    root.minsize(500, 200)
    timer = PomodoroTimer(root)
    root.mainloop()
