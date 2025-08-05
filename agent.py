from pendulum import Pendulum
import random
from collections import deque
from model import Linear_QNet, QTrainer
import numpy as np
import torch
from helper import plot

max_memory = 100_000
batch = 1000
lr = 0.001

class Agent():
    def __init__(self):
        self.n_sims = 0
        self.epsilon = 0
        self.gamma = 0.8
        self.memory = deque(maxlen=max_memory)
        self.model = Linear_QNet(8, 128, 2)
        self.trainer = QTrainer(self.model, lr, self.gamma)
    
    def normalize(self, value, min_val, max_val):
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def get_state(self, sim):
        s = sim.stem
        j = sim.joint
        state = [
            self.normalize(s.x, 90, 1110),
            self.normalize(s.vec_x, -35.0, 35.0),
            self.normalize(s.force_vec_x, -2.0, 2.0),
            self.normalize(j.x, 90.0, 1110.0),
            self.normalize(j.y, 120.0, 520.0),
            self.normalize(j.vec_x, -35.0, 35.0),
            self.normalize(j.vec_y, -35.0, 35.0),
            self.normalize(j.force_vec_x, -10.0, 10.0)
        ]
        return np.array(state, dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_lm(self):
        if len(self.memory) > batch:
            mini_sample = random.sample(self.memory, batch)
        else:
            mini_sample = self.memory
        state, action, reward, next_state, done = zip(*mini_sample)
        self.trainer.train_step(state, action, reward, next_state, done)
    def train_sm(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 500 - self.n_sims
        final_move = [0, 0]
        if random.randint(0, 250) < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def train():
    plot_time = []
    plot_mean_time = []
    total_time = 0
    record = 0
    agent = Agent()
    sim = Pendulum()
    while True:
        state_old = agent.get_state(sim)
        move = agent.get_action(state_old)
        reward, done, time_up = sim.sim_step(move)
        state_new = agent.get_state(sim)
        agent.train_sm(state_old, move, reward, state_new, done)
        agent.remember(state_old, move, reward, state_new, done)
        if done:
            sim.reset()
            agent.n_sims += 1
            agent.train_lm()
            if time_up > record:
                record = time_up
                agent.model.save()
            print("Sim", agent.n_sims, "Record: ", record)
            plot_time.append(time_up)
            total_time += time_up
            mean_score = total_time/ agent.n_sims
            plot_mean_time.append(mean_score)
            plot(plot_time, plot_mean_time)

if __name__ == "__main__":
    train()