import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import models


class PPONetsIG(nn.Module):
    def __init__(self,
                 act_dim,
                 device,
                 fix_cnn=False,
                 rnn_type='lstm',
                 rnn_hidden_dim=128,
                 rnn_num=1):
        super().__init__()
        # Input image size: [80, 80, 3] and [80, 80, 3]
        self.device = device
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num = rnn_num

        num_states = 7

        # TODO choose action space and distribution
        # TODO Gaussians std learnable

        self.large_map_resnet_model = models.resnet18(pretrained=True)
        self.small_map_resnet_model = models.resnet18(pretrained=True)
        resnet_models = [self.large_map_resnet_model,
                         self.small_map_resnet_model]
        if fix_cnn:
            for model in resnet_models:
                for param in model:
                    param.requires_grad = False
        num_ftrs = self.large_map_resnet_model.fc.in_features
        num_in = 0

        self.large_map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.large_map_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128
        self.small_map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.small_map_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128

        self.state_encoder = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ELU(),
        )
        num_in += 128

        self.merge_fc = nn.Linear(num_in, rnn_hidden_dim)
        if rnn_type == 'gru':
            rnn_cell = nn.GRU
        elif rnn_type == 'lstm':
            rnn_cell = nn.LSTM
        else:
            raise ValueError('unsupported rnn type: %s' % rnn_type)
        self.rnn = rnn_cell(input_size=rnn_hidden_dim,
                            hidden_size=rnn_hidden_dim,
                            num_layers=rnn_num)
        self.actor_fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ELU(),
        )
        self.actor_head = nn.Linear(32, act_dim)
        self.critic_fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ELU(),
        )
        self.critic_head = nn.Linear(32, 1)

        # TODO init distributions
        self.log_std = nn.Parameter(torch.ones(act_dim))

        self.reset_parameters()
        print('========= requires_grad =========')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('=================================')
        print('****************************')
        print('MAP as INPUT')
        print('****************************')

    def forward(self, large_maps, small_maps, states,
                hidden_state=None, action=None, deterministic=False, expert_action=None):
        seq_len, batch_size, C, H, W = large_maps.size()
        large_maps = large_maps.view(batch_size * seq_len, C, H, W)
        l_cnn_out = self.large_map_resnet_model(large_maps)
        l_cnn_out = l_cnn_out.view(seq_len, batch_size, -1)

        seq_len, batch_size, C, H, W = small_maps.size()
        small_maps = small_maps.view(batch_size * seq_len, C, H, W)
        s_cnn_out = self.small_map_resnet_model(small_maps)
        s_cnn_out = s_cnn_out.view(seq_len, batch_size, -1)

        seq_len, batch_size, dims = states.size()
        states = states.view(batch_size * seq_len, dims)
        st_fc_out = self.state_encoder(states)
        st_fc_out = st_fc_out.view(seq_len, batch_size, -1)

        cnn_out = torch.cat((l_cnn_out, s_cnn_out, st_fc_out), dim=-1)

        rnn_in = F.elu(self.merge_fc(cnn_out))

        rnn_out, hidden_state = self.rnn(rnn_in, hidden_state)
        pi = self.actor_head(self.actor_fc(rnn_out))
        val = self.critic_head(self.critic_fc(rnn_out))

        # dist = Categorical(logits=pi)
        dist = MultivariateNormal(loc=pi, covariance_matrix=torch.diag(self.log_std))
        if action is None:
            if not deterministic:
                action = dist.sample()
            else:
                # action = torch.max(pi, dim=2)[1]
                action = dist.mean
        log_prob = dist.log_prob(action)

        expert_log_prob = dist.log_prob(expert_action) if expert_action is not None else None

        return action, log_prob, dist.entropy(), val, hidden_state, pi, expert_log_prob

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.rnn_num,
                                batch_size,
                                self.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.rnn_num,
                                batch_size,
                                self.rnn_hidden_dim).to(self.device))
        else:
            return torch.zeros(self.rnn_num,
                               batch_size,
                               self.rnn_hidden_dim).to(self.device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias.data, 0)
