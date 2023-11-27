# code adapted from https://github.com/wendelinboehmer/dcg
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
import rware

env = gym.make("rware-medium-4ag-v1")

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args["hidden_dim"])
        if self.args["use_rnn"]:
            self.rnn = nn.GRUCell(args["hidden_dim"], args["hidden_dim"])
        else:
            self.rnn = nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.fc2 = nn.Linear(args["hidden_dim"], args["n_actions"])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args["hidden_dim"]).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args["hidden_dim"])
        if self.args["use_rnn"]:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h


args = {"hidden_dim": 64, "n_actions": 5, "use_rnn": False}
agent = RNNAgent(75, args)
agent.load_state_dict(th.load("C:/Users/Clover/Documents/My Received Files/3-3-JCB/marl/marl/four/1/agent.th", map_location=th.device('cpu')))
hs = agent.init_hidden()
env.reset() 
obs = env.get_obs()
obs = th.tensor([obs])
for _ in range(1000):
    action, hs = agent(obs, hs)
    obs, reward, done, info = env.step(actions)

env.close()