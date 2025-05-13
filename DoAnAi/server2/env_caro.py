import numpy as np

class CaroEnv:
    def __init__(self, board_size=15, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 1: agent, -1: opponent
        self.done = False
        return self.board.copy()

    def step(self, action):
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0 or self.done:
            return self.board.copy(), -1, True, {}  # Phạt nếu chọn ô đã có

        self.board[row, col] = self.current_player

        if self.check_win(row, col, self.current_player):
            self.done = True
            reward = 1 if self.current_player == 1 else -1
        elif np.all(self.board != 0):
            self.done = True
            reward = 0.5  # Draw
        else:
            reward = 0
            self.current_player *= -1

        return self.board.copy(), reward, self.done, {}

    def check_win(self, row, col, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for sign in [-1, 1]:
                r, c = row, col
                while True:
                    r += sign * dr
                    c += sign * dc
                    if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_length:
                return True
        return False

    def available_actions(self):
        return [r * self.board_size + c for r in range(self.board_size) for c in range(self.board_size) if self.board[r, c] == 0]
