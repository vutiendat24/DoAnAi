import numpy as np
import tensorflow as tf
import random
from collections import deque
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
BOARD_SIZE = 15  # Kích thước bàn cờ 15x15
EMPTY = 0  # Ô trống
PLAYER_X = 1  # Người chơi X
PLAYER_O = -1  # Người chơi O (AI)
WINNING_LENGTH = 5  # Chiều dài để chiến thắng

# Cài đặt tham số cho DQN
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Tỷ lệ khám phá ban đầu
EPSILON_END = 0.1  # Tỷ lệ khám phá tối thiểu
EPSILON_DECAY = 0.9999  # Tỷ lệ giảm của epsilon
LEARNING_RATE = 0.0001
TARGET_UPDATE_FREQUENCY = 1000  # Tần suất cập nhật mạng đích
MINIMAX_DEPTH = 2  # Độ sâu của thuật toán minimax

# Phần thưởng
REWARD_WIN = 1000
REWARD_DRAW = 10
REWARD_LOSE = -1000

class GomokuEnv:
    """Môi trường cờ caro."""
    
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        """Khởi tạo lại trạng thái ban đầu của môi trường."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = PLAYER_X
        self.done = False
        self.winner = None
        self.last_move = None
        return self.get_state()
    def set_board(self, board):
        self.board = np.array(board)
        self.current_player = -1  # Đảm bảo là lượt AI

    def get_state(self):
        # Nếu bạn dùng 3 kênh (AI, người, lượt) thì chuyển từ self.board
        # Giả sử return về state dạng np.array (shape: [3, 15, 15])
        player_board = (self.board == self.current_player).astype(np.float32)
        opp_board = (self.board == -self.current_player).astype(np.float32)
        turn_channel = np.full_like(player_board, self.current_player, dtype=np.float32)
        return np.stack([player_board, opp_board, turn_channel], axis=0)
    def get_state(self):
        """Lấy trạng thái hiện tại của môi trường."""
        # Chuyển đổi bảng thành định dạng phù hợp cho mạng neural
        # Player X: kênh 1, Player O: kênh 2, Empty: kênh 3
        state = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        state[:, :, 0] = (self.board == PLAYER_X)
        state[:, :, 1] = (self.board == PLAYER_O)
        state[:, :, 2] = (self.board == EMPTY)
        return state
    
    def get_valid_moves(self):
        """Trả về danh sách các nước đi hợp lệ."""
        if self.done:
            return []
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == EMPTY]
    
    def is_valid_move(self, move):
        """Kiểm tra xem nước đi có hợp lệ không."""
        i, j = move
        return 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i, j] == EMPTY
    
    def step(self, move):
        """Thực hiện nước đi và trả về trạng thái mới, phần thưởng và trạng thái kết thúc."""
        if self.done:
            return self.get_state(), 0, True, None
        
        i, j = move
        if not self.is_valid_move(move):
            return self.get_state(), -100, False, None  # Phạt nặng cho nước đi không hợp lệ
        
        self.board[i, j] = self.current_player
        self.last_move = move
        
        # Kiểm tra chiến thắng
        if self._check_win(i, j):
            self.done = True
            self.winner = self.current_player
            reward = REWARD_WIN if self.current_player == PLAYER_O else REWARD_LOSE
            return self.get_state(), reward, True, {"winner": self.winner}
        
        # Kiểm tra hòa
        if len(self.get_valid_moves()) == 0:
            self.done = True
            self.winner = None
            return self.get_state(), REWARD_DRAW, True, {"winner": None}
        
        # Tính điểm tức thời dựa trên nước đi
        immediate_reward = self._calculate_immediate_reward(i, j)
        
        # Đổi người chơi
        self.current_player = -self.current_player
        
        return self.get_state(), immediate_reward, False, None
    
    def _check_win(self, i, j):
        """Kiểm tra chiến thắng sau nước đi tại vị trí (i, j)."""
        player = self.board[i, j]
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for di, dj in directions:
            count = 1  # Đã có 1 quân tại vị trí hiện tại
            
            # Đếm về một phía
            ni, nj = i + di, j + dj
            while 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == player:
                count += 1
                ni += di
                nj += dj
            
            # Đếm về phía đối diện
            ni, nj = i - di, j - dj
            while 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == player:
                count += 1
                ni -= di
                nj -= dj
            
            if count >= WINNING_LENGTH:
                return True
                
        return False
    
    def _calculate_immediate_reward(self, i, j):
        """
        Tính điểm tức thời cho nước đi dựa trên bảng điểm được cung cấp.
        Điểm này phản ánh sự đánh giá chiến thuật của nước đi.
        """
        player = self.board[i, j]
        reward = 0
        
        # Tính điểm cho các tình huống khác nhau
        patterns = self._evaluate_board_patterns(i, j, player)
        
        # Phần thưởng dựa trên bảng điểm đã cung cấp
        if patterns["open_four"]:  # Tạo hàng 4 không bị chặn
            reward += 100 if player == PLAYER_O else -100
        elif patterns["half_open_four"]:  # Tạo hàng 4 bị chặn một đầu
            reward += 80 if player == PLAYER_O else -80
        elif patterns["open_three"]:  # Tạo hàng 3 không bị chặn
            reward += 70 if player == PLAYER_O else -70
        elif patterns["half_open_three"]:  # Tạo hàng 3 bị chặn một đầu
            reward += 8 if player == PLAYER_O else -8
        elif patterns["open_two"]:  # Tạo hàng 2 không bị chặn
            reward += 3 if player == PLAYER_O else -3
        elif patterns["half_open_two"]:  # Tạo hàng 2 bị chặn một đầu
            reward += 1 if player == PLAYER_O else -1
            
        # Chặn các mối đe dọa của đối thủ
        patterns_opponent = self._evaluate_opponent_threats(i, j, -player)
        
        if patterns_opponent["double_threat"]:  # Chặn nước đôi
            reward += 100 if player == PLAYER_O else -100
        if patterns_opponent["triple_threat"]:  # Chặn nước ba
            reward += 20 if player == PLAYER_O else -20
        if patterns_opponent["open_four"]:  # Chặn hàng 4 không bị chặn
            reward += 100 if player == PLAYER_O else -100
        if patterns_opponent["half_open_four"]:  # Chặn hàng 4 bị chặn một đầu
            reward += 80 if player == PLAYER_O else -80
        if patterns_opponent["open_three"]:  # Chặn hàng 3 không bị chặn
            reward += 80 if player == PLAYER_O else -80
        if patterns_opponent["half_open_three"]:  # Chặn hàng 3 bị chặn một đầu
            reward += 10 if player == PLAYER_O else -10
            
        # Kiểm tra nước đi tạo cơ hội cho đối thủ
        potential_threats = self._evaluate_potential_opponent_threats(i, j, -player)
        
        if potential_threats["double_threat"]:  # Đối thủ có thể tạo nước đôi
            reward -= 20 if player == PLAYER_O else 20
        if potential_threats["open_four"]:  # Đối thủ có thể tạo hàng 4 không bị chặn
            reward -= 100 if player == PLAYER_O else 100
            
        # Tạo nước đôi (2 đường thắng tiềm năng)
        if self._count_winning_paths(i, j, player) >= 2:
            reward += 50 if player == PLAYER_O else -50
            
        # Tạo nước ba (3 đường thắng tiềm năng)
        if self._count_double_threat_paths(i, j, player) >= 3:
            reward += 25 if player == PLAYER_O else -25
        
        return reward if player == PLAYER_O else -reward  # DQN học từ quan điểm của PLAYER_O (AI)
    
    def _evaluate_board_patterns(self, i, j, player):
        """Đánh giá các mẫu (pattern) trên bàn cờ sau nước đi tại (i, j)."""
        patterns = {
            "open_four": False,
            "half_open_four": False,
            "open_three": False,
            "half_open_three": False,
            "open_two": False,
            "half_open_two": False
        }
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for di, dj in directions:
            # Xác định chuỗi quân cờ theo hướng hiện tại
            sequence, borders = self._get_sequence(i, j, di, dj, player)
            length = len(sequence)
            
            # Kiểm tra các mẫu dựa trên chiều dài và biên
            if length == 4:
                if borders[0] and borders[1]:  # Cả hai đầu đều mở
                    patterns["open_four"] = True
                elif borders[0] or borders[1]:  # Một đầu mở
                    patterns["half_open_four"] = True
            elif length == 3:
                if borders[0] and borders[1]:
                    patterns["open_three"] = True
                elif borders[0] or borders[1]:
                    patterns["half_open_three"] = True
            elif length == 2:
                if borders[0] and borders[1]:
                    patterns["open_two"] = True
                elif borders[0] or borders[1]:
                    patterns["half_open_two"] = True
        
        return patterns
    
    def _evaluate_opponent_threats(self, i, j, opponent):
        """Đánh giá các mối đe dọa của đối thủ bị chặn bởi nước đi tại (i, j)."""
        # Giả sử đối thủ đã đi vào vị trí này, đánh giá các mối đe dọa
        original_value = self.board[i, j]
        self.board[i, j] = opponent
        
        patterns = self._evaluate_board_patterns(i, j, opponent)
        patterns["double_threat"] = (self._count_winning_paths(i, j, opponent) >= 2)
        patterns["triple_threat"] = (self._count_double_threat_paths(i, j, opponent) >= 3)
        
        # Khôi phục giá trị ban đầu
        self.board[i, j] = original_value
        
        return patterns
    
    def _evaluate_potential_opponent_threats(self, i, j, opponent):
        """Đánh giá các mối đe dọa tiềm năng mà đối thủ có thể tạo ra ở nước đi tiếp theo."""
        potential_threats = {
            "double_threat": False,
            "open_four": False
        }
        
        # Lưu trạng thái hiện tại
        original_value = self.board[i, j]
        
        # Với mỗi nước đi tiếp theo có thể của đối thủ
        for ni in range(self.board_size):
            for nj in range(self.board_size):
                if self.board[ni, nj] == EMPTY and (ni != i or nj != j):
                    # Thử nước đi
                    self.board[ni, nj] = opponent
                    
                    # Kiểm tra nếu đối thủ có thể tạo nước đôi hoặc hàng 4 không bị chặn
                    if self._count_winning_paths(ni, nj, opponent) >= 2:
                        potential_threats["double_threat"] = True
                    
                    patterns = self._evaluate_board_patterns(ni, nj, opponent)
                    if patterns["open_four"]:
                        potential_threats["open_four"] = True
                    
                    # Hoàn tác
                    self.board[ni, nj] = EMPTY
                    
                    # Nếu đã tìm thấy cả hai loại mối đe dọa, thoát
                    if potential_threats["double_threat"] and potential_threats["open_four"]:
                        self.board[i, j] = original_value
                        return potential_threats
        
        # Khôi phục giá trị ban đầu
        self.board[i, j] = original_value
        
        return potential_threats
    
    def _get_sequence(self, i, j, di, dj, player):
        """Lấy chuỗi quân cờ liên tiếp theo một hướng cụ thể."""
        sequence = [self.board[i, j]]
        
        # Kiểm tra xem các đầu của chuỗi có mở không
        left_open = False
        right_open = False
        
        # Kiểm tra đầu trái
        ni, nj = i - di, j - dj
        if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
            if self.board[ni, nj] == EMPTY:
                left_open = True
        
        # Mở rộng sang phải
        ni, nj = i + di, j + dj
        while 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == player:
            sequence.append(self.board[ni, nj])
            ni += di
            nj += dj
        
        # Kiểm tra đầu phải
        if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
            if self.board[ni, nj] == EMPTY:
                right_open = True
        
        return sequence, (left_open, right_open)
    
    def _count_winning_paths(self, i, j, player):
        """Đếm số lượng đường thắng tiềm năng từ vị trí (i, j)."""
        count = 0
        original_value = self.board[i, j]
        self.board[i, j] = player
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for di, dj in directions:
            # Đếm số quân liên tiếp và kiểm tra biên
            sequence, borders = self._get_sequence(i, j, di, dj, player)
            
            # Nếu đủ dài và có ít nhất một đầu mở, đó là một đường thắng tiềm năng
            if len(sequence) >= WINNING_LENGTH or (len(sequence) == WINNING_LENGTH - 1 and (borders[0] or borders[1])):
                count += 1
        
        self.board[i, j] = original_value
        return count
    
    def _count_double_threat_paths(self, i, j, player):
        """Đếm số lượng đường có thể tạo thành nước đôi từ vị trí (i, j)."""
        count = 0
        original_value = self.board[i, j]
        self.board[i, j] = player
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for di, dj in directions:
            # Đếm số quân liên tiếp và kiểm tra biên
            sequence, borders = self._get_sequence(i, j, di, dj, player)
            
            # Nếu có 2 quân liên tiếp và cả hai đầu đều mở, đó là một đường có thể tạo nước đôi
            if len(sequence) == 2 and borders[0] and borders[1]:
                count += 1
        
        self.board[i, j] = original_value
        return count
    
    def render(self):
        """Hiển thị bàn cờ."""
        symbols = {EMPTY: '.', PLAYER_X: 'X', PLAYER_O: 'O'}
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                row.append(symbols[self.board[i, j]])
            print(' '.join(row))
        print()

class DQNAgent:
    """AI sử dụng thuật toán Deep Q-Network."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # (board_size, board_size, 3)
        self.action_size = action_size  # board_size^2
        
        # Bộ nhớ replay
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        # Tỷ lệ khám phá
        self.epsilon = EPSILON_START
        
        # Xây dựng mạng chính và mạng đích
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Đếm số bước để cập nhật mạng đích
        self.target_update_counter = 0
    
    def _build_model(self):
        """Xây dựng mạng neural cho DQN."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                                   input_shape=self.state_size),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu trữ trải nghiệm vào bộ nhớ replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves, minimax_depth=0):
        """Chọn hành động dựa trên epsilon-greedy hoặc kết hợp với minimax."""
        if len(valid_moves) == 0:
            return None
        
        
        # Khai thác: sử dụng mạng DQN hoặc kết hợp với minimax
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        
        if minimax_depth > 0:
            # Kết hợp với minimax
            # Đầu tiên chọn top k nước đi theo q_values
            k = min(3, len(valid_moves))  # Chọn top 3 hoặc ít hơn nếu không đủ
            valid_moves_indices = [(i * BOARD_SIZE + j) for i, j in valid_moves]
            valid_q_values = [(idx, q_values[idx]) for idx in valid_moves_indices]
            top_moves = sorted(valid_q_values, key=lambda x: x[1], reverse=True)[:k]
            
            # Đánh giá top k nước đi bằng minimax
            best_move = None
            best_value = float('-inf')
            
            for idx, _ in top_moves:
                move = (idx // BOARD_SIZE, idx % BOARD_SIZE)
                board_copy = np.copy(self.env.board)
                self.env.board[move[0], move[1]] = PLAYER_O
                
                value = self._minimax(board_copy, minimax_depth, False, float('-inf'), float('inf'))
                self.env.board = np.copy(board_copy)  # Khôi phục bàn cờ
                
                if value > best_value:
                    best_value = value
                    best_move = move
            
            return best_move
        else:
            # Chỉ sử dụng DQN
            valid_moves_indices = [(i * BOARD_SIZE + j) for i, j in valid_moves]
            valid_q_values = [q_values[idx] for idx in valid_moves_indices]
            max_index = np.argmax(valid_q_values)
            return valid_moves[max_index]
    
    def _minimax(self, board, depth, is_maximizing, alpha, beta):
        """Thuật toán minimax kết hợp alpha-beta pruning."""
        # Kiểm tra điều kiện dừng
        if depth == 0:
            return self._evaluate_board(board)
        
        # Lấy danh sách các nước đi hợp lệ
        valid_moves = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i, j] == EMPTY]
        
        if len(valid_moves) == 0:
            return 0  # Hòa
        
        if is_maximizing:
            max_eval = float('-inf')
            for i, j in valid_moves:
                board[i, j] = PLAYER_O
                if self._check_win_minimax(board, i, j):
                    board[i, j] = EMPTY
                    return 1000  # AI thắng
                
                eval = self._minimax(board, depth - 1, False, alpha, beta)
                board[i, j] = EMPTY
                max_eval = max(max_eval, eval)
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            for i, j in valid_moves:
                board[i, j] = PLAYER_X
                if self._check_win_minimax(board, i, j):
                    board[i, j] = EMPTY
                    return -1000  # Người chơi thắng
                
                eval = self._minimax(board, depth - 1, True, alpha, beta)
                board[i, j] = EMPTY
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            return min_eval
    
    def _check_win_minimax(self, board, i, j):
        """Kiểm tra chiến thắng cho thuật toán minimax."""
        player = board[i, j]
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for di, dj in directions:
            count = 1  # Đã có 1 quân tại vị trí hiện tại
            
            # Đếm về một phía
            ni, nj = i + di, j + dj
            while 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni, nj] == player:
                count += 1
                ni += di
                nj += dj
            
            # Đếm về phía đối diện
            ni, nj = i - di, j - dj
            while 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni, nj] == player:
                count += 1
                ni -= di
                nj -= dj
            
            if count >= WINNING_LENGTH:
                return True
                
        return False
    
    def _evaluate_board(self, board):
        """Đánh giá giá trị của bàn cờ cho minimax."""
        # Đánh giá dựa trên các mẫu trên bàn cờ
        score = 0
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == PLAYER_O:  # AI (tối đa)
                    score += self._evaluate_position(board, i, j, directions, PLAYER_O)
                elif board[i, j] == PLAYER_X:  # Người chơi (tối thiểu)
                    score -= self._evaluate_position(board, i, j, directions, PLAYER_X)
        
        return score
    
    def _evaluate_position(self, board, i, j, directions, player):
        """Đánh giá giá trị của một vị trí cụ thể trên bàn cờ."""
        score = 0
        
        for di, dj in directions:
            # Đếm số quân liên tiếp theo hướng (di, dj)
            count = 1  # Quân cờ hiện tại
            open_ends = 0  # Số đầu mở
            
            # Kiểm tra một đầu
            ni, nj = i - di, j - dj
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni, nj] == EMPTY:
                open_ends += 1
            
            # Đếm về phía khác
            ni, nj = i + di, j + dj
            while 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni, nj] == player:
                count += 1
                ni += di
                nj += dj
            
            # Kiểm tra đầu còn lại
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni, nj] == EMPTY:
                open_ends += 1
            
            # Tính điểm dựa trên số quân liên tiếp và số đầu mở
            if count >= 5:  # Thắng
                score += 10000
            elif count == 4:
                if open_ends == 2:  # Hàng 4 không bị chặn
                    score += 1000
                elif open_ends == 1:  # Hàng 4 bị chặn một đầu
                    score += 100
            elif count == 3:
                if open_ends == 2:  # Hàng 3 không bị chặn
                    score += 50
                elif open_ends == 1:  # Hàng 3 bị chặn một đầu
                    score += 10
            elif count == 2:
                if open_ends == 2:  # Hàng 2 không bị chặn
                    score += 5
                elif open_ends == 1:  # Hàng 2 bị chặn một đầu
                    score += 1
        
        return score
    
    def replay(self, batch_size):
        """Huấn luyện mô hình từ bộ nhớ replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            i, j = action
            action_idx = i * BOARD_SIZE + j
            if not done:
                
                next_state_expanded = np.expand_dims(next_state, axis=0)
                target = reward + GAMMA * np.amax(self.target_model.predict(next_state_expanded, verbose=0)[0])
            
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action_idx] = target
            
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        
        # Giảm epsilon theo thời gian
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            
        # Cập nhật mạng đích
        self.target_update_counter += 1
        if self.target_update_counter >= TARGET_UPDATE_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def load(self, name):
        """Tải mô hình từ file."""
        self.model.load_weights(name)
        self.target_model.load_weights(name)
        
    def save(self, name):
        """Lưu mô hình vào file."""
        self.model.save_weights(name)

class HybridAgent:
    """AI sử dụng kết hợp DQN và Minimax."""
    
    def __init__(self, env):
        self.env = env
        self.state_size = (BOARD_SIZE, BOARD_SIZE, 3)
        self.action_size = BOARD_SIZE * BOARD_SIZE
        self.dqn_agent = DQNAgent(self.state_size, self.action_size)
        self.dqn_agent.env = env  # Gán môi trường cho DQN Agent để sử dụng trong minimax
    
    def act(self, state, valid_moves):
        """Chọn hành động sử dụng kết hợp DQN và Minimax."""
        # Ở những nước đi đầu tiên, sử dụng DQN thuần túy
        if len([p for p in self.env.board.flatten() if p != EMPTY]) < 2:
            return self.dqn_agent.act(state, valid_moves, minimax_depth=MINIMAX_DEPTH)
        else:
            # Khi trò chơi đã diễn ra một thời gian, sử dụng kết hợp với minimax
            return self.dqn_agent.act(state, valid_moves, minimax_depth=0)
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu trải nghiệm vào bộ nhớ replay của DQN."""
        self.dqn_agent.remember(state, action, reward, next_state, done)
    
    def replay(self, batch_size):
        """Huấn luyện mô hình DQN."""
        self.dqn_agent.replay(batch_size)
    
    def load(self, name):
        """Tải mô hình từ file."""
        self.dqn_agent.load(name)
        
    def save(self, name):
        """Lưu mô hình vào file."""
        self.dqn_agent.save(name)

def train_agent(episodes=5000, render_every=500, save_every=1000):
    """Huấn luyện agent qua nhiều episode."""
    env = GomokuEnv()
    agent = HybridAgent(env)
    
    # Tạo tên file dựa trên thời điểm huấn luyện
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f"gomoku_dqn_{timestamp}"
    
    # Theo dõi performance qua các episode
    scores = []
    win_count = 0
    episode_times = []
    
    # Bắt đầu quá trình huấn luyện
    for episode in range(1, episodes + 1):
        start_time = time.time()
        
        # Khởi tạo lại môi trường
        state = env.reset()
        
        # Biến để tính tổng reward trong một episode
        total_reward = 0
        
        # Vòng lặp chính của episode
        while not env.done:
            # AI (PLAYER_O) chọn hành động
            if env.current_player == PLAYER_O:
                valid_moves = env.get_valid_moves()
                action = agent.act(state, valid_moves)
                if action is None:
                    break
                
                next_state, reward, done, info = env.step(action)
                
                # Lưu trải nghiệm vào bộ nhớ replay
                agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            else:
                # Người chơi (PLAYER_X) - ở đây chúng ta sử dụng nước đi ngẫu nhiên để huấn luyện
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                
                action = random.choice(valid_moves)
                next_state, _, done, _ = env.step(action)
                state = next_state
        
        # Huấn luyện mô hình sau mỗi episode
        agent.replay(BATCH_SIZE)
        
        # Tính toán thời gian cho episode
        end_time = time.time()
        episode_time = end_time - start_time
        episode_times.append(episode_time)
        
        # Ghi nhận thông tin của episode
        scores.append(total_reward)
        if env.winner == PLAYER_O:
            win_count += 1
        
        # In thông tin của episode
        if episode % 10 == 0:
            print(f"Episode: {episode}/{episodes}, Score: {total_reward:.2f}, Win Rate: {win_count/episode:.2f}, Epsilon: {agent.dqn_agent.epsilon:.4f}")
        
        # Render trò chơi mỗi vài episode
        if episode % render_every == 0:
            print(f"\nEpisode {episode}:")
            env.render()
            
        # Lưu mô hình mỗi vài episode
        if episode % save_every == 0:
            agent.save(f"{model_filename}_ep{episode}.weights.h5")
            
            # Vẽ biểu đồ hiệu suất
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(scores)
            plt.title('Rewards per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            
            plt.subplot(1, 3, 2)
            plt.plot([i/j for i, j in enumerate(range(1, episode + 1))], label='Win Rate')
            plt.title('Win Rate')
            plt.xlabel('Episode')
            plt.ylabel('Rate')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(episode_times)
            plt.title('Training Time per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Time (s)')
            
            plt.tight_layout()
            plt.savefig(f"{model_filename}_performance_ep{episode}.png")
            plt.close()
    
    # Lưu mô hình cuối cùng
    agent.save(f"{model_filename}_final.weights.h5")
    
    return agent, scores

def play_against_ai(agent, player_first=True):
    """Chơi cờ Gomoku với AI đã được huấn luyện."""
    env = GomokuEnv()
    state = env.reset()
    
    print("Bàn cờ Gomoku 15x15:")
    print("Người chơi: X, AI: O")
    env.render()
    
    while not env.done:
        if (env.current_player == PLAYER_X and player_first) or (env.current_player == PLAYER_O and not player_first):
            # Lượt của người chơi
            valid = False
            while not valid:
                try:
                    move_str = input("Nhập nước đi của bạn (hàng cột, ví dụ '7 8'): ")
                    i, j = map(int, move_str.split())
                    action = (i, j)
                    if env.is_valid_move(action):
                        valid = True
                    else:
                        print("Nước đi không hợp lệ. Vui lòng thử lại.")
                except ValueError:
                    print("Định dạng không hợp lệ. Vui lòng nhập 'hàng cột', ví dụ '7 8'.")
            
            next_state, _, done, _ = env.step(action)
            state = next_state
        else:
            # Lượt của AI
            print("AI đang suy nghĩ...")
            valid_moves = env.get_valid_moves()
            action = agent.act(state, valid_moves)
            
            if action is None:
                break
                
            print(f"AI đã đi: {action[0]} {action[1]}")
            next_state, _, done, _ = env.step(action)
            state = next_state
        
        print("\nBàn cờ hiện tại:")
        env.render()
    
    # Kết thúc trò chơi
    if env.winner == PLAYER_X:
        print("Chúc mừng! Bạn đã thắng!")
    elif env.winner == PLAYER_O:
        print("AI đã thắng. Chúc may mắn lần sau!")
    else:
        print("Trò chơi kết thúc hòa.")

def main():
    """Hàm chính để chạy chương trình."""
    print("Gomoku AI với Deep Q-Network và Minimax")
    print("1. Huấn luyện AI mới")
    print("2. Tải AI đã huấn luyện và chơi")
    
    choice = input("Chọn chế độ (1-2): ")
    
    if choice == '1':
        episodes = int(input("Nhập số lượng episode để huấn luyện (mặc định: 5000): ") or "5000")
        agent, _ = train_agent(episodes=episodes)
        
        play_choice = input("Bạn có muốn chơi với AI vừa huấn luyện không? (y/n): ")
        if play_choice.lower() == 'y':
            player_first = input("Bạn có muốn đi trước không? (y/n): ").lower() == 'y'
            play_against_ai(agent, player_first)
    
    elif choice == '2':
        # model_path = input("Nhập đường dẫn đến file mô hình (ví dụ: gomoku_dqn_YYYYMMDD-HHMMSS_final.hweights.h5): ")
        model_path="D:\\Study\\Nam3\\HocKy2\\AI\\DOAN\\server111\\gomoku_dqn_final.weights.h5"
        
        env = GomokuEnv()
        agent = HybridAgent(env)
        
        try:
            agent.load(model_path)
            print("Đã tải mô hình thành công!")
            
            player_first = input("Bạn có muốn đi trước không? (y/n): ").lower() == 'y'
            play_against_ai(agent, player_first)
        except:
            print("Không thể tải mô hình. Vui lòng kiểm tra lại đường dẫn.")
    
    else:
        print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
