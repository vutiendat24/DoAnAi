import numpy as np
import torch
from env_caro import CaroEnv
from dqn_agent import DQNAgent

def train():
    # Khởi tạo môi trường và agent
    env = CaroEnv()
    n_actions = env.board_size * env.board_size  # Tổng số hành động có thể (15x15 = 225)
    agent = DQNAgent((1, env.board_size, env.board_size), n_actions)

    # Số lượng vòng lặp huấn luyện
    n_episodes = 150
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 3000

    try:
        for episode in range(n_episodes):
            state = env.reset()
            state = state[np.newaxis, :, :]  # Thêm chiều cho trạng thái để phù hợp với mô hình
            total_reward = 0
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)

            for step in range(1000):
                available_actions = env.available_actions()  # Lấy các hành động hợp lệ
                action = agent.select_action(state, epsilon, available_actions)  # Chọn hành động

                next_state, reward, done, _ = env.step(action)  # Thực hiện hành động và nhận thông tin phản hồi
                next_state = next_state[np.newaxis, :, :]  # Thêm chiều cho trạng thái tiếp theo

                agent.store_transition(state, action, reward, next_state, done)  # Lưu trữ quá trình chuyển tiếp
                agent.train_step()  # Huấn luyện agent

                state = next_state
                total_reward += reward

                if done:
                    break

            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

            # Lưu mô hình định kỳ (mỗi 100 episodes) để tránh mất dữ liệu
            if episode % 100 == 0 and episode > 0:
                torch.save(agent.model.state_dict(), f"dqn_caro_episode_{episode}.pth")
                print(f"Model saved at episode {episode} as dqn_caro_episode_{episode}.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        # Lưu mô hình khi dừng
        torch.save(agent.model.state_dict(), "dqn_caro_interrupt.pth")
        print("Model saved as dqn_caro_interrupt.pth")
        return

    # Lưu mô hình cuối cùng nếu hoàn thành
    torch.save(agent.model.state_dict(), "dqn_caro_final.pth")
    print("Training completed and model saved as dqn_caro_final.pth")

if __name__ == "__main__":
    train()