import random
import math

# import gymnasium as gym
import gym
go_env = gym.make('gym_go:go-v0', size=19, komi=0, reward_method='real')

class Node:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def expand(self):
        for action in go_env.valid_moves():
            
            print(go_env.valid_moves())

            state, reward, done, info = go_env.step(action)
            child_node = Node(go_env, parent=self, action=action)
            self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == len(go_env.valid_moves())

    def best_child(self, exploration_constant):
        return max(self.children, key=lambda c: c.value/c.visits + exploration_constant * math.sqrt(2*math.log(self.visits)/c.visits))

    def rollout(self):
        state = self.state
        while not go_env.game_ended():
            action = random.choice(go_env.uniform_random_action())
            state = go_env.step(action)
        return go_env.winner()

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

def mcts(go_env, max_iterations, exploration_constant=math.sqrt(2)):
    root_node = Node(go_env)
    for _ in range(max_iterations):
        node = tree_policy(root_node, exploration_constant)
        result = node.rollout()
        node.backpropagate(result)
    best_child_node = max(root_node.children, key=lambda c: c.visits)
    return best_child_node.action

def tree_policy(node, exploration_constant):
    while not node.env.game_ended():
        if not node.is_fully_expanded():
            node.expand()
            return node.children[-1]
        else:
            node = node.best_child(exploration_constant)
    return node



first_action = (2,5)
second_action = (5,2)
go_env.reset()
# state, reward, done, info = go_env.step(first_action)
# state, reward, done, info = go_env.step(second_action)
# go_env.render('terminal')

go_env.reset()  # Replace with your game state class
max_iterations = 1000

while not go_env.game_ended():
    action = mcts(go_env, max_iterations)
    game_state = go_env.step(action)
