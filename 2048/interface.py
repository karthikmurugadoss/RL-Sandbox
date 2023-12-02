"""
Inspired and adapted from:
https://github.com/kiteco/python-youtube-code/tree/master/build-2048-in-python
"""
import time
import tkinter as tk

import colors
from engine import GameEngine


class GameInterface(tk.Frame):
    KEY_TO_ACTION_LABEL = {
        "<Left>": "left",
        "<Right>": "right",
        "<Up>": "up",
        "<Down>": "down",
        "<KeyPress-u>": "undo",
    }

    def __init__(self, engine, use_agent=False):
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("2048")
        self.main_grid = tk.Frame(
            self, bg=colors.GRID_COLOR, bd=3, width=400, height=400
        )
        self.main_grid.grid(pady=(80, 0))
        self.make_GUI()
        self.update_GUI(engine.matrix, engine.score, engine.num_moves)

        if not use_agent:
            for key, action_label in self.KEY_TO_ACTION_LABEL.items():
                self.master.bind(key, self.get_action_handle(engine, action_label))
            self.mainloop()
        else:
            self.update()
            # self.delay()

    @staticmethod
    def delay():
        time.sleep(4)

    def make_GUI(self):
        """"""
        # Grid
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.main_grid, bg=colors.EMPTY_CELL_COLOR, width=100, height=100
                )
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg=colors.EMPTY_CELL_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)

        # Score
        score_frame = tk.Frame(self)
        score_frame.place(relx=0.33, y=40, anchor="center")
        tk.Label(score_frame, text="Score", font=colors.SCORE_LABEL_FONT).grid(row=0)
        self.score_label = tk.Label(score_frame, text="0", font=colors.SCORE_FONT)
        self.score_label.grid(row=1)

        # Move counter
        move_frame = tk.Frame(self)
        move_frame.place(relx=0.66, y=40, anchor="center")
        tk.Label(move_frame, text="Moves", font=colors.SCORE_LABEL_FONT).grid(row=0)
        self.move_label = tk.Label(move_frame, text="0", font=colors.SCORE_FONT)
        self.move_label.grid(row=1)

    def update_GUI(self, matrix, score, num_moves):
        """"""
        # update grid
        for i in range(4):
            for j in range(4):
                cell_value = matrix[i][j]
                if cell_value == 0:
                    self.cells[i][j]["frame"].configure(bg=colors.EMPTY_CELL_COLOR)
                    self.cells[i][j]["number"].configure(
                        bg=colors.EMPTY_CELL_COLOR, text=""
                    )
                else:
                    self.cells[i][j]["frame"].configure(
                        bg=colors.CELL_COLORS[cell_value]
                    )
                    self.cells[i][j]["number"].configure(
                        bg=colors.CELL_COLORS[cell_value],
                        fg=colors.CELL_NUMBER_COLORS[cell_value],
                        font=colors.CELL_NUMBER_FONTS[cell_value],
                        text=str(cell_value),
                    )

        # update score
        self.score_label.configure(text=score)
        self.move_label.configure(text=num_moves)
        self.update()

    def get_action_handle(self, engine, action_label):
        """"""

        def action(event):
            matrix, _, num_moves = engine.action(action_label)
            self.update_GUI(matrix, engine.score, num_moves)

        return action


engine = GameEngine()
# GameInterface(engine)

# breakpoint()

# quit()

# NOTE: Temporary code - Will remove later
game_interface = GameInterface(engine)  # , use_agent=True)

quit()

import random


def run():
    action_label = random.choice(["left", "right", "up", "down"])
    matrix, score, num_moves = engine.action(action_label)
    game_interface.update_GUI(matrix, score, num_moves)
    print(engine.matrix)
    # game_interface.delay()
    game_interface.after(4000, run)


game_interface.after(4000, run)
game_interface.mainloop()
