
from typing import Callable, Tuple
from abc import ABC, abstractmethod, abstractproperty
import torch as pt
from ..constants import DEFAULT_TENSOR_TYPE


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def compute_returns(rewards: pt.Tensor, gamma: float = 0.99) -> pt.Tensor:
    n_steps = len(rewards)
    discounts = pt.logspace(0, n_steps-1, n_steps, gamma)
    returns = [(discounts[:n_steps-t] * rewards[t:]).sum()
               for t in range(n_steps)]
    return pt.tensor(returns)


def compute_gae(rewards: pt.Tensor, values: pt.Tensor, gamma: float = 0.99, lam: float = 0.97) -> pt.Tensor:
    n_steps = len(rewards)
    factor = pt.logspace(0, n_steps-1, n_steps, gamma*lam)
    delta = rewards[:-1] + gamma * values[1:] - values[:-1]
    gae = [(factor[:n_steps-t-1] * delta[t:]).sum()
           for t in range(n_steps - 1)]
    return pt.tensor(gae)


class FCPolicy(pt.nn.Module):
    def __init__(self, n_states: int, n_actions: int, action_min: pt.Tensor,
                 action_max: pt.Tensor, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        super(FCPolicy, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._action_min = action_min
        self._action_max = action_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        #self._layers.append(pt.nn.Linear(self._n_neurons, 2*self._n_actions))
        self._last_layer = pt.nn.Linear(self._n_neurons, 2*self._n_actions)

    @pt.jit.ignore
    def _scale(self, actions: pt.Tensor) -> pt.Tensor:
        return (actions - self._action_min) / (self._action_max - self._action_min)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for layer in self._layers:
            x = self._activation(layer(x))
        return 1.0 + pt.nn.functional.softplus(self._last_layer(x))

    @pt.jit.ignore
    def predict(self, states: pt.Tensor, actions: pt.Tensor) -> pt.Tensor:
        out = self.forward(states)
        c0 = out[:, :self._n_actions]
        c1 = out[:, self._n_actions:]
        beta = pt.distributions.Beta(c0, c1)
        if len(actions.shape) == 1:
            scaled_actions = self._scale(actions.unsqueeze(-1))
        else:
            scaled_actions = self._scale(actions)
        log_p = beta.log_prob(scaled_actions)
        if len(actions.shape) == 1:
            return log_p.squeeze(), beta.entropy().squeeze()
        else:
            return log_p.sum(1), beta.entropy().sum(1)


class FCValue(pt.nn.Module):
    def __init__(self, n_states: int, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        super(FCValue, self).__init__()
        self._n_states = n_states
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up value network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        self._layers.append(pt.nn.Linear(self._n_neurons, 1))

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for i_layer in range(len(self._layers) - 1):
            x = self._activation(self._layers[i_layer](x))
        return self._layers[-1](x).squeeze()


class Attention(pt.nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear1 = pt.nn.Linear(num_inputs, 2) 
        pt.nn.init.kaiming_normal_(self.linear1.weight)

    def forward(self, x):
        return self.linear1(x)


class AttentionValue(pt.nn.Module):
    def __init__(self, n_states: int, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu, dim_E: int = 20):
        super(AttentionValue, self).__init__()
        self._n_states = n_states
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation
        self._dim_E = dim_E
        
        # Layer E
        self.linearE = pt.nn.Linear(self._n_states, self._dim_E) 
        # attention layers
        self.attention_layers = pt.nn.ModuleList(
            [Attention(self._dim_E) for _ in range(self._n_states)]
        )
        # set up value network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        self._layers.append(pt.nn.Linear(self._n_neurons, 1))
        
        # init layer
        # for layer in self._layers:
        #     if isinstance(layer, pt.nn.Linear):
        #         pt.nn.init.kaiming_normal_(layer.weight)
        #         pt.nn.init.zeros_(layer.bias)
        # pt.nn.init.kaiming_normal_(self.linearE.weight)

    def forward(self, states):
        # Layer E
        E = pt.tanh(self.linearE(states))
        # attention layers
        attention_out_list = []
        for i in range(self._n_states):
            attention_FC = self.attention_layers[i](E)
            attention_out = pt.nn.functional.softmax(attention_FC, dim=1)
            attention_out_list.append(attention_out[:, 1])
        Attention = pt.stack(attention_out_list).T
        state_attention = pt.mul(states, Attention)
        # value network
        for i_layer in range(len(self._layers) - 1):
            x = self._activation(self._layers[i_layer](state_attention))
        return self._layers[-1](x).squeeze(), Attention
        
class AttentionPolicy(pt.nn.Module):
    def __init__(self, n_states: int, n_actions: int, action_min: pt.Tensor,
                 action_max: pt.Tensor, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu, dim_E: int = 20):
        super(AttentionPolicy, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._action_min = action_min
        self._action_max = action_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation
        self._dim_E = dim_E

        # Layer E
        self.linearE = pt.nn.Linear(self._n_states, self._dim_E) 
        # attention layers
        self.attention_layers = pt.nn.ModuleList(
            [Attention(self._dim_E) for _ in range(self._n_states)]
        )
        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        #self._layers.append(pt.nn.Linear(self._n_neurons, 2*self._n_actions))
        self._last_layer = pt.nn.Linear(self._n_neurons, 2*self._n_actions)

        #weight init
        for layer in self._layers:
            if isinstance(layer, pt.nn.Linear):
                pt.nn.init.kaiming_normal_(layer.weight)
                pt.nn.init.zeros_(layer.bias)
        pt.nn.init.kaiming_normal_(self._last_layer.weight)
        pt.nn.init.zeros_(self._last_layer.bias)
        pt.nn.init.kaiming_normal_(self.linearE.weight)

    @pt.jit.ignore
    def _scale(self, actions: pt.Tensor) -> pt.Tensor:
        return (actions - self._action_min) / (self._action_max - self._action_min)

    @pt.jit.ignore
    def compute_Attention_matrix(self, states):
        # Layer E
        E = pt.tanh(self.linearE(states))
        # attention layers
        attention_out_list = []
        #for i in range(self._n_states):
        for i, layer in enumerate(self.attention_layers):
            # attention_FC = self.attention_layers[i](E)
            attention_FC = layer(E)
            attention_out = pt.nn.functional.softmax(attention_FC, dim=1)
            attention_out_list.append(attention_out[:, 1])
        Attention = pt.stack(attention_out_list).T
        state_attention = pt.mul(states, Attention)
        return Attention

    def forward(self, states) -> pt.Tensor:
        # Layer E
        E = pt.tanh(self.linearE(states))
        #print(E.shape)
        # attention layers
        attention_out_list = []
        #for i in range(self._n_states):
        print("len attention layers:", len(self.attention_layers))
        for i, layer in enumerate(self.attention_layers):
            # attention_FC = self.attention_layers[i](E)
            attention_FC = layer(E)
            attention_out = pt.nn.functional.softmax(attention_FC, dim=1)
            attention_out_list.append(attention_out[:, 1])
        Attention = pt.stack(attention_out_list).T
        print("Attention shape:", Attention.shape)
        state_attention = pt.mul(states, Attention)
        # policy network

        #x = self._activation(self._layers[0](state_attention))

        #print("len self._layers:", len(self._layers))
        #hidden_layers_list = self._layers[1:]
        #print("len self._layers[1:]:", len(self._layers[1:])

        # for i, layer in enumerate(self._layers):
        #     if i == 0:
        #         x = self._activation(layer(state_attention))
        #     else:

        x = state_attention
        for layer in self._layers:
            x = self._activation(layer(x))
        return 1.0 + pt.nn.functional.softplus(self._last_layer(x))

    @pt.jit.ignore
    def predict(self, states: pt.Tensor, actions: pt.Tensor):
        out = self.forward(states) # change BC if forward returns second tensor
        c0 = out[:, :self._n_actions]
        c1 = out[:, self._n_actions:]
        beta = pt.distributions.Beta(c0, c1)
        if len(actions.shape) == 1:
            scaled_actions = self._scale(actions.unsqueeze(-1))
        else:
            scaled_actions = self._scale(actions)
        log_p = beta.log_prob(scaled_actions)
        if len(actions.shape) == 1:
            return log_p.squeeze(), beta.entropy().squeeze()
        else:
            return log_p.sum(1), beta.entropy().sum(1)

class Agent(ABC):
    """Common interface for all agents.
    """

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def trace_policy(self):
        pass

    @abstractproperty
    def history(self):
        pass
