from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from dqn_agent import DQNAgent
from env_caro import CaroEnv

app = FastAPI()

# Cấu hình CORS để cho phép frontend React (localhost:3000) gửi request tới server
origins = [
    "http://localhost:5173",  # Địa chỉ của frontend React
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Cho phép frontend từ localhost:3000
    allow_credentials=True,
    allow_methods=["POST","GET","PUT","DELETE","OPTION"],  # Cho phép tất cả các phương thức HTTP (GET, POST, ...)
    allow_headers=["*"],  # Cho phép tất cả các header
)

# Khởi tạo môi trường Caro
env = CaroEnv()
n_actions = env.board_size * env.board_size
agent = DQNAgent((1, env.board_size, env.board_size), n_actions)

# Load model đã train
model_path = "dqn_caro_final.pth"
agent.model.load_state_dict(torch.load(model_path))
agent.model.eval()  # Chế độ dự đoán

class BoardState(BaseModel):
    board: list[list[int]]  # Bàn cờ 15x15

@app.post("/predict")
async def predict_move(board_state: BoardState):
    board = np.array(board_state.board)

    # Đảm bảo đầu vào có kích thước đúng
    if board.shape != (env.board_size, env.board_size):
        return {"error": "Bàn cờ phải có kích thước 15x15"}

    state = board[np.newaxis, :, :]  # Thêm chiều batch

    # Lọc các hành động có sẵn (ô chưa bị đánh)
    available_actions = [action for action in env.available_actions() if board[action // env.board_size][action % env.board_size] == 0]
    
    if not available_actions:  # Nếu không còn ô trống
        return {"error": "Không còn nước đi hợp lệ"}

    # Dự đoán nước đi tiếp theo từ các hành động có sẵn
    action = agent.select_action(state, epsilon=0.0, available_actions=available_actions)

    # Chuyển action thành (row, col)
    row, col = divmod(action, env.board_size)

    return {"row": row, "col": col}

@app.get("/")
def read_root():
    return {"message": "Welcome to Caro AI API!"}
