import random
import numpy as np
import pygame
import torch
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque


STUDENT_ID = 'a1234567'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=10000)  # experience replay buffer (D in Algorithm 2)
        self.action_list = []  # to store ordered list of valid actions

        # Assuming state has 4 features (e.g., bird position, velocity, etc.) and 2 actions (flap or not)
        self.network = MLPRegression(input_dim=4, output_dim=2, learning_rate=0.001)
        self.network2 = MLPRegression(input_dim=4, output_dim=2, learning_rate=0.001)

        # initialise Q_f's parameter by Q's
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0          # ε for exploration, decayed over time
        self.n = 32                 # batch size for experience replay
        self.discount_factor = 0.99  # γ in Algorithm 2

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)
            
    def build_state(self, state: dict) -> np.ndarray:
        """
        Convert the state dictionary into a feature vector.
        Args:
            state: input state representation (the state dictionary from the game environment)
        Returns:
            phi_t: feature vector (numpy array)
        """
        bird = state.get('bird', {})
        pipe = state.get('pipe', {})

        bird_y = bird.get('y', 0.0)
        bird_velocity = bird.get('velocity', 0.0)
        pipe_x = pipe.get('x', 0.0)
        pipe_height = pipe.get('height', 0.0)

        # Convert to numpy feature vector
        phi_t = np.array([bird_y, bird_velocity, pipe_x, pipe_height], dtype=np.float32)
        return phi_t

    
    def choose_action(self, state: dict, action_table: dict) -> int:
        phi_t = self.build_state(state)

        # Exclude 'quit_game' from learning
        if not self.action_list:
            self.action_list = [action_table['do_nothing'], action_table['jump']]

        if self.mode == 'train' and np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(self.action_list))
        else:
            q_values = self.network.predict(phi_t.reshape(1, -1))
            action_index = np.argmax(q_values)

        action_code = self.action_list[action_index]

        if self.mode == 'train':
            self.storage.append([phi_t, action_index, None, None])  # only 0 or 1

        return action_code

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        if self.mode != 'train' or len(self.storage) == 0:
            return

        # Get the last stored transition (phi_t, a_t, _, _)
        phi_t, a_t, _, _ = self.storage[-1]

        # Build next state representation
        phi_t_next = self.build_state(state)

        # Calculate reward
        reward = state.get('score', 0)
        if state.get('done', False):
            if state.get('score', 0) >= state.get('target_score', float('inf')):
                reward += 10
            else:
                reward -= 10

        # Update the transition
        self.storage[-1] = (phi_t, a_t, reward, phi_t_next)

        # Only train if enough samples
        if len(self.storage) >= self.n:
            batch = random.sample(self.storage, self.n)

            # Extract from batch
            phi_batch = np.vstack([t[0] for t in batch])
            a_batch = [t[1] for t in batch]
            r_batch = [t[2] for t in batch]
            q_next_batch = np.array([self.network2.predict(t[3]) for t in batch], dtype=np.float32)

            # Build action mapping (action_code → index)
            if isinstance(action_table, dict):
                action_values = sorted(action_table.values())
            else:
                action_values = sorted(action_table)

            action_to_index = {v: i for i, v in enumerate(action_values)}
            action_indices = np.array([action_to_index.get(a, 0) for a in a_batch], dtype=np.int32)

            # Predict Q(s, ·)
            q_pred = self.network.predict(phi_batch)
            targets = q_pred.copy()
            for i in range(self.n):
                targets[i, action_indices[i]] = r_batch[i] + self.discount_factor * np.max(q_next_batch[i])

            # Build weight mask: only update chosen action
            weights = np.zeros_like(targets)
            for i in range(self.n):
                weights[i, action_indices[i]] = 1.0

            # Train using fit_step (no direct model access)
            self.network.fit_step(phi_batch, targets, weights)

            # Epsilon decay
            self.epsilon = max(0.1, self.epsilon * 0.95)



    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    episodes = 10000
    for episode in range(episodes):
        env.play(player=agent)

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)

        # store the best model based on your judgement
        agent.save_model(path='my_model.ckpt')

        # you'd want to clear the memory after one or a few episodes
        ...

        # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
        ...

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
