import math
import numpy as np
import tensorflow as tf
from typing import List, Dict, NamedTuple

class Action(NamedTuple):
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

class Player:
    def __init__(self, id: int):
        self.id = id

class TicTacToeConfig:
    def __init__(self):
        self.board_size = 3
        self.action_space_size = 9
        self.num_simulations = 50
        self.max_moves = 9
        self.discount = 1.0
        self.dirichlet_alpha = 0.3
        self.num_training_steps = 1000
        self.batch_size = 32
        self.td_steps = 9
        self.num_unroll_steps = 3
        self.lr_init = 0.001
        self.lr_decay_steps = 1000
        self.visit_softmax_temperature_fn = lambda _: 1.0

    def new_game(self):
        return TicTacToeGame(self)

class TicTacToeGame:
    def __init__(self, config: TicTacToeConfig):
        self.config = config
        self.board = np.zeros((config.board_size, config.board_size), dtype=int)
        self.current_player = Player(1)
        self.move_count = 0

    def terminal(self) -> bool:
        return self.winner() is not None or self.move_count == self.config.max_moves

    def winner(self) -> Player:
        for player in [1, 2]:
            # Check rows and columns
            for i in range(3):
                if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                    return Player(player)
            # Check diagonals
            if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
                return Player(player)
        return None

    def legal_actions(self) -> List[Action]:
        return [Action(i // 3, i % 3) for i in range(9) if self.board[i // 3, i % 3] == 0]

    def apply(self, action: Action):
        self.board[action.row, action.col] = self.current_player.id
        self.current_player = Player(3 - self.current_player.id)  # Switch player
        self.move_count += 1

    def make_image(self):
        return self.board.copy()

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

class Network(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.value_head = tf.keras.layers.Dense(1, activation='tanh')
        self.policy_head = tf.keras.layers.Dense(9, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

def train_network(config: TicTacToeConfig):
    network = Network()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr_init)

    for _ in range(config.num_training_steps):
        batch = generate_batch(config)
        loss = update_weights(optimizer, network, batch)
        print(f"Loss: {loss.numpy()}")

def generate_batch(config: TicTacToeConfig):
    batch = []
    for _ in range(config.batch_size):
        game = config.new_game()
        game_history = []

        while not game.terminal():
            root = Node(0)
            action = select_action(config, game, root, network)
            game_history.append((game.make_image(), action))
            game.apply(action)

        winner = game.winner()
        value = 0 if winner is None else (1 if winner.id == 1 else -1)

        for state, action in game_history:
            batch.append((state, action, value))
            value = -value  # Flip value for opponent's turn

    return batch

def select_action(config: TicTacToeConfig, game: TicTacToeGame, root: Node, network: Network):
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        current_game = game.copy()

        while node.expanded():
            action, node = select_child(config, node)
            current_game.apply(action)
            search_path.append(node)

        value, policy = network(np.expand_dims(current_game.make_image(), 0))
        value = value.numpy()[0, 0]
        policy = policy.numpy()[0]

        expand_node(node, current_game.legal_actions(), policy)
        backpropagate(search_path, value, game.current_player)

    return select_child(config, root)[0]

def select_child(config: TicTacToeConfig, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child

def ucb_score(config: TicTacToeConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.action_space_size + 1) / config.action_space_size) + config.dirichlet_alpha
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = -child.value()
    return prior_score + value_score

def expand_node(node: Node, actions: List[Action], policy):
    for action in actions:
        node.children[action] = Node(policy[action.row * 3 + action.col])

def backpropagate(search_path: List[Node], value: float, to_play: Player):
    for node in reversed(search_path):
        node.value_sum += value if to_play.id == 1 else -value
        node.visit_count += 1
        value = -value

def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch):
    with tf.GradientTape() as tape:
        loss = 0
        for state, action, target_value in batch:
            value, policy = network(np.expand_dims(state, 0))
            loss += tf.square(target_value - value)
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=[[action.row * 3 + action.col]], logits=policy)

    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    return loss

# Main training loop
config = TicTacToeConfig()
train_network(config)