import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal, MultivariateNormal
from models.cnn_encoders import CNN3Layer, CNN3Layer_old, ResNetEnc

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

        num_states = 6

        # TODO choose action space and distribution
        # TODO Gaussians std learnable

        self.large_map_encoder = ResNetEnc(img_size=80, img_ch=3, out_ftrs=128)
        self.small_map_encoder = ResNetEnc(img_size=80, img_ch=3, out_ftrs=128)
        cnn_models = [self.large_map_encoder,
                         self.small_map_encoder]
        if fix_cnn:
            for model in cnn_models:
                for name, param in model.named_parameters():
                    if name not in ['network.fc.weight', 'network.fc.bias']:
                        param.requires_grad = False
                    else:
                        continue

        self.state_encoder = nn.Sequential(
            nn.Linear(num_states, 128),
            # nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.Linear(32, 128),
        )
        num_in = 3*128

        self.merge_fc = nn.Sequential(
            nn.Linear(num_in, rnn_hidden_dim),
            nn.ELU(),
        )
        if rnn_type is not None:
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
        self.log_std = nn.Parameter(torch.zeros(act_dim, requires_grad=True))

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

        l_cnn_out = self.large_map_encoder(large_maps)

        s_cnn_out = self.small_map_encoder(small_maps)

        # seq_len, batch_size, dims = states.size()
        # states = states.reshape(batch_size * seq_len, dims)
        st_fc_out = self.state_encoder(states)
        # st_fc_out = st_fc_out.reshape(seq_len, batch_size, -1)

        cnn_out = torch.cat((l_cnn_out, s_cnn_out, st_fc_out), dim=-1)

        rnn_in = self.merge_fc(cnn_out)

        if self.rnn_type is not None:
            rnn_out, hidden_state = self.rnn(rnn_in, hidden_state)
        else:
            rnn_out = rnn_in

        pi = self.actor_head(self.actor_fc(rnn_out))
        val = self.critic_head(self.critic_fc(rnn_out))

        # dist = Categorical(logits=pi)
        std = torch.exp(self.log_std)
        dist = Independent(Normal(loc=pi, scale=std), 1)
        if action is None:
            if not deterministic:
                action = dist.sample()
            else:
                # action = torch.max(pi, dim=2)[1]
                action = dist.mean
        log_prob = dist.log_prob(action)

        expert_log_prob = dist.log_prob(expert_action) if expert_action is not None else None

        return action, log_prob, dist.entropy(), val, hidden_state.detach(), pi, expert_log_prob

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
