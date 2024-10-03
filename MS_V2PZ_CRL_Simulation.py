import argparse
import math
import time
import warnings
from random import randint

from tqdm import trange, tqdm

from ParallelPZEnv import simulation_env
from str2bool import str2bool as strtobool
import numpy as np
from numpy import min as nmin
from numpy import max as nmax
from numpy import mean, quantile, median, std
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import os
import scipy.stats
from welford import Welford
from ParallelPZEnv import n_timesteps, n_cars
import inequalipy as ineq


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Run Settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    # Algorithm Run Settings
    parser.add_argument("--num-episodes", type=int, default=20000,
        help="total episodes of the experiments")
    parser.add_argument("--num-cars", type=int, default=500, nargs="?", const=True,
                        help="number of cars in experiment")
    parser.add_argument("--random-cars", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, num cars is ignored and we will sample a p_dist for number of cars at each epoch.")
    parser.add_argument("--network-size", type=int, choices=[1,3,4,5], default=5,
        help="the size of the network. 1: small, 2: medium, 3: large, 4: extra large")


    # parser.add_argument("--num-steps", type=int, default=((n_timesteps+3)*5),
    parser.add_argument("--num-steps", type=int, default=5500,
    help = "the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")

    # Args made by me
    # Agent params
    parser.add_argument("--reward-clip", type=float, default=0,
        help="The value to clip the reward. If left at 0, the reward will not be clipped")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    # Env params
    # parser.add_argument("--fixed-road-cost", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Sets the initial road cost to be fixed. If not set, road cost will be random at each episode.")
    parser.add_argument("--linear-arrival-dist", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Sets the arrival dist to be linear. If not set, arrival dist will be beta dist.")
    parser.add_argument("--normalised-observations", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Normalises the agent's observations. If not set, observations will be raw values.")
    parser.add_argument("--rewardfn", type=str, default="MaxProfit", nargs="?", const=True,
        help="reward function for agent to use. has to be predefined.")
    parser.add_argument("--action-masks", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether invalid actions are masked or not.")

    parser.add_argument("--reward-norm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether raw rewards are used or rewards are normalised.")
    parser.add_argument("--warmup-arn", type= lambda x: bool(strtobool(x)), default = False, nargs = "?", const = True,
        help = "Toggle whether we warmup the reward normalisation by playing the environment randomly before training agent.")

    parser.add_argument("--batch-return-norm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles whether raw rewards are used or rewards are normalised per episode batch.")
    parser.add_argument("--batch-obs-norm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles whether raw obs are used or obs are normalised per episode batch.")

    # parser.add_argument("--actor-spect-norm", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggle whether we use spectral norm between layers in the actor.")
    # parser.add_argument("--critic-spect-norm", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggle whether we use spectral norm between layers in the critic.")

    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    # parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.1,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.1,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--eps_per_update", type=int, default=64, help="Number of episodes per update. If 1, same behaviour as before.")
    args = parser.parse_args()
    args.num_envs = 2
    # args.batch_size = int(args.num_envs * args.num_steps)
    args.batch_size = 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_steps = ((args.num_steps + 1) * 5)
    args.total_timesteps = n_timesteps * args.num_episodes
    # fmt: on
    return args

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)



class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        if args.network_size == 1:
            self.critic = nn.Sequential(
                layer_init(nn.Linear((12), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear((12), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 3), std=0.01),
            )
        elif args.network_size == 2:
            self.critic = nn.Sequential(
                layer_init(nn.Linear((12), 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear((12), 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 3), std=0.01),
            )
        elif args.network_size == 3:
            self.critic = nn.Sequential(
                layer_init(nn.Linear((12), 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear((12), 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 512)),
                nn.Tanh(),
                layer_init(nn.Linear(512, 3), std=0.01),
            )
        elif args.network_size == 4:
            self.critic = nn.Sequential(
                layer_init(nn.Linear((12), 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear((12), 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 3), std=0.01),
            )
        elif args.network_size == 5:
            self.critic = nn.Sequential(
                layer_init(nn.Linear((12), 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.BatchNorm1d(1024),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.BatchNorm1d(1024),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.BatchNorm1d(1024),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear((12), 1024)),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.BatchNorm1d(1024),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.BatchNorm1d(1024),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 1024)),
                nn.BatchNorm1d(1024),
                nn.Tanh(),
                layer_init(nn.Linear(1024, 3), std=0.01),
            )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        # return self.critic(self.network(x / 255.0))
        return self.critic(x)

    # def get_action_and_value(self, x, action=None):
    #     hidden = self.network(x / 255.0)
    #     logits = self.actor(hidden)
    #     probs = Categorical(logits=logits)
    #     if action is None:
    #         action = probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    def get_action_and_value(self, x, action=None, action_masks=None):
        x = x.float()
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        if not args.action_masks:
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        else:
            if action_masks is None:
                raise Exception("Action Masks called but none passed to action and value funct.")
            probs = CategoricalMasked(logits=logits, masks=action_masks)
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    # breakpoint()
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)
    # obs = obs.flatten()
    # convert to torch
    obs = torch.tensor(obs).to(device)
    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    # breakpoint()
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm


if __name__ == "__main__":
    args = parse_args()
    print("----------- RUN DETAILS -----------")
    print("N CARS:", args.num_cars if not args.random_cars else "Random")
    print("N TIMESTEPS", n_timesteps)
    print("-----------      END     -----------")
    run_name = f"MMRP_Online__5EpisodesPerRollOut:MinimiseTT__{n_timesteps}__{n_cars}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project="cleanRL",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps else "cpu")

    """ ENV SETUP """
    env = simulation_env(
        initial_road_cost="Fixed",
        fixed_road_cost=50.0,
        arrival_dist="Linear" if args.linear_arrival_dist else "Beta",
        normalised_obs=True if args.normalised_observations else False,
        road0_capacity=15,
        road0_fftraveltime=20,
        road1_capacity=30,
        road1_fftraveltime=20,
        reward_fn=args.rewardfn,
        n_car=args.num_cars
    )
    # max_cycles = env.timesteps * 10
    num_agents = 2
    num_actions = 3
    observation_size = env.observation_space(env.possible_agents[0]).shape

    if args.reward_norm:
        agent_welford = {agt: Welford() for agt in env.possible_agents}


    if args.warmup_arn:
        if not args.reward_norm:
            warnings.warn("ARN is set to False but Warmup ARN is set to true. Warmup ARN will have no impact. Is this what you meant to do?")
        else:
            obs, infos = env.reset()
            print("Warming up ARN with random actions")
            for step in range(0, n_timesteps):
                actions = {
                    agent: env.action_space(agent).sample() for agent in env.agents
                }
                observations, rewards, terms, truncs, infos = env.step(actions)
                for agt in env.agents:
                    agent_welford[agt].add(np.array([rewards[env.agent_name_mapping[agt]]]))
                if any([terms[a] for a in terms]):
                    print("Warm up complete.")
                    break

    """ LEARNER SETUP """
    agent = Agent().to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)
    lr = args.learning_rate
    optimizer = optim.Adam(
        [
            {'params': agent.actor.parameters(), 'lr': lr, 'eps': 1e-5},
            {'params': agent.critic.parameters(), 'lr': lr, 'eps': 1e-5}
        ]
    )
    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0

    rb_obs = torch.zeros(((n_timesteps+1)*args.eps_per_update, num_agents) + (12,), dtype=torch.float32).to(
        device
    )
    rb_actions = torch.zeros(((n_timesteps+1)*args.eps_per_update, num_agents), dtype=torch.float32).to(
        device
    )
    rb_logprobs = torch.zeros(((n_timesteps+1)*args.eps_per_update, num_agents), dtype=torch.float32).to(
        device
    )
    rb_rewards = torch.zeros(((n_timesteps+1)*args.eps_per_update+1, num_agents), dtype=torch.float32).to(
        device
    )
    rb_terms = torch.zeros(((n_timesteps+1)*args.eps_per_update +1, num_agents), dtype=torch.float32).to(device)
    rb_values = torch.zeros(((n_timesteps+1)*args.eps_per_update +1, num_agents), dtype=torch.float32).to(
        device
    )
    rb_action_masks = torch.zeros(((n_timesteps+2)*args.eps_per_update, num_agents, num_actions), dtype=torch.float32).to(
        device
    )

    num_updates = args.total_timesteps // args.batch_size
    start_time = time.time()

    completed_eps = 0

    """ TRAINING LOGIC """
    # train for n number of episodes
    # pbar = tqdm(range(args.num_episodes))
    global_step = 0
    while completed_eps < args.num_episodes:
        end_step = 0
        end_steps = []
        batch_step = 0

        if args.anneal_lr:
            frac = 1.0 - (completed_eps / args.num_episodes)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[1]["lr"] = lrnow

        batch_episodic_return = []
        batch_episodic_difference = []
        batch_n_cars = []

        # collect number of episodes
        for episode in range(args.eps_per_update):
            with torch.no_grad():
                next_obs, info = env.reset(seed=None, random_cars=args.random_cars)
                total_episodic_return = 0
                for step in range(0, n_timesteps+1):
                    # rb_action_masks[batch_step] = torch.tensor([False, False, False])
                    global_step += 1
                    obs = batchify_obs(next_obs, device)
                    if args.action_masks:
                        s_masks = torch.tensor([info[agt]['action_mask'] for agt in env.agents])
                        actions, logprobs, _, values = agent.get_action_and_value(obs, action_masks=s_masks)
                    else:
                        actions, logprobs, _, values = agent.get_action_and_value(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = env.step(
                        unbatchify(actions, env)
                    )
                    if args.reward_norm:
                        for agt in env.agents:
                            agent_welford[agt].add(np.array([rewards[env.agent_name_mapping[agt]]]))
                            if global_step < 2 or agent_welford[agt].var_s[0] == 0.0:
                                pass
                            else:
                                rewards[env.agent_name_mapping[agt]] = (rewards[env.agent_name_mapping[agt]] -
                                                                        agent_welford[agt].mean[0]) / math.sqrt(
                                    agent_welford[agt].var_s[0] + 1e-8)

                    if args.reward_clip != 0:
                        rewards = {
                            agt: min(max(r, -args.reward_clip), args.reward_clip)
                            for agt, r in rewards.items()
                        }

                    rb_obs[batch_step] = obs
                    rb_rewards[batch_step] = batchify(rewards, device)
                    rb_terms[batch_step] = batchify(terms, device)
                    rb_actions[batch_step] = actions
                    rb_logprobs[batch_step] = logprobs
                    rb_values[batch_step] = values.flatten()
                    if args.action_masks:
                        rb_action_masks[batch_step] = s_masks

                    # compute episodic return
                    total_episodic_return += rb_rewards[batch_step].cpu().numpy()
                    batch_step += 1
                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        # print("total episodic return: ", total_episodic_return, sum(total_episodic_return))
                        end_step = batch_step
                        end_steps.append(end_step)
                        completed_eps += 1
                        batch_episodic_return.append(sum(total_episodic_return))
                        batch_episodic_difference.append(total_episodic_return[0] - total_episodic_return[1])
                        batch_n_cars.append(env.n_cars)
                        break

        print(f"At Episode {completed_eps}: Batch episodic return: {np.mean(batch_episodic_return)}, BER/Car: {np.mean(batch_episodic_return) / n_cars}")
        print(f"At Episode {completed_eps}: Batch episodic difference: {np.mean(batch_episodic_difference)}, BED/Car: {np.mean(batch_episodic_difference) / n_cars}")
        if args.random_cars:
            print(f"Batch n_cars: {batch_n_cars}")

        # lets see if batchwise-reward-norm will work
        # pbar.update(args.eps_per_update)
        # if args.batch_return_norm:
        #     b_return_mean = rb_rewards.mean(axis=0)
        #     b_return_std = rb_rewards.std(axis=0)
        #     print(b_return_mean, b_return_std)
        #     rb_rewards = (rb_rewards - b_return_mean) / b_return_std

        """
        # bootstrap value if not done
        with torch.no_grad():
            for end_step in end_steps:
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(end_step)):
                    delta = (
                            rb_rewards[t]
                            + args.gamma * rb_values[t + 1] * (1 - rb_terms[t + 1])
                            - rb_values[t]
                    )
                    # rb_advantages[t] = delta + args.gamma * args.gamma * rb_advantages[t + 1]
                    # added in Generalised Advantage Estimation. Hopefully this smooths the learning process
                    rb_advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * (1 - rb_terms[t + 1]) * lastgaelam
                    )
                rb_returns = rb_advantages + rb_values
        """
        # FIXME: this could work. this could not work. who knows.
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(end_step)):
                delta = (
                        rb_rewards[t]
                        + args.gamma * rb_values[t + 1] * (1 - rb_terms[t + 1])
                        - rb_values[t]
                )
                rb_advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * (1 - rb_terms[t + 1]) * lastgaelam
                )
            rb_returns = rb_advantages + rb_values
        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:batch_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:batch_step+1], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:batch_step+1], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:batch_step+1], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:batch_step+1], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:batch_step+1], start_dim=0, end_dim=1)
        if args.action_masks:
            b_action_masks = torch.flatten(rb_action_masks[:batch_step+1], start_dim=0, end_dim=1)
        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))

        # lets see if batchwise-reward-norm will work
        # if args.batch_return_norm:
        #     b_return_mean = b_returns.mean()
        #     b_return_std = b_returns.std()
        #     b_returns = (b_returns - b_return_mean) / b_return_std
        # if args.batch_obs_norm:
        #     b_obs = (b_obs - b_obs.mean(axis=0)) / b_obs.std(axis=0)

        # b_inds = np.arange(args.batch_size)
        clip_fracs = []
        for repeat in range(args.update_epochs):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            # for start in range(0, len(b_obs), args.batch_size):
            for start in range(0, len(b_obs), args.batch_size):
                # select the indices we want to train on
                end = start + args.batch_size
                batch_index = b_index[start:end]
                if args.action_masks:
                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index], action_masks=b_action_masks[batch_index]
                    )
                else:
                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                )

                # Policy loss
                # pg_loss1 = -b_advantages[batch_index] * ratio
                # pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                #     ratio, 1 - args.clip_coef, 1 + args.clip_coef
                # )
                # Policy loss using normalised advantages
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()

                # # Value loss
                # value = value.flatten()
                # v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                # v_clipped = b_values[batch_index] + torch.clamp(
                #     value - b_values[batch_index],
                #     -clip_coef,
                #     clip_coef,
                # )
                # v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                # v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                # v_loss = 0.5 * v_loss_max.mean()
                if args.clip_vloss:
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -args.clip_coef,
                        args.clip_coef,
                        )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((value - b_returns[batch_index]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                # print("Value loss:", v_loss, ", Policy Loss", pg_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # print("Value loss:", v_loss.item(), ", Policy Loss", pg_loss.item())
            # for name, mdl in zip(['Actor', 'Critic'], [agent.actor, agent.critic]):
            #     grad_norm = get_gradient_norm(mdl)
            #     print(f"{name} Gradient Norm: {grad_norm}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/agent_0_advantage", rb_advantages[:, 0].mean().item(), global_step)
        writer.add_scalar("losses/agent_1_advantage", rb_advantages[:, 1].mean().item(), global_step)
        writer.add_scalar("losses/agent_0_logprobs", rb_logprobs[:, 0].mean().item(), global_step)
        writer.add_scalar("losses/agent_1_logprobs", rb_logprobs[:, 1].mean().item(), global_step)
        writer.add_scalar("losses/agent_0_returns_means", rb_returns[:, 0].mean().item(), global_step)
        writer.add_scalar("losses/agent_1_returns_means", rb_returns[:, 1].mean().item(), global_step)
        writer.add_scalar("losses/agent_0_returns_vars", rb_returns[:, 0].var().item(), global_step)
        writer.add_scalar("losses/agent_1_returns_vars", rb_returns[:, 1].var().item(), global_step)
        for name, mdl in zip(['Actor', 'Critic'], [agent.actor, agent.critic]):
            grad_norm = get_gradient_norm(mdl)
            # print(f"{name} Gradient Norm: {grad_norm}")
            writer.add_scalar(f"losses/{name}_grad_norm", grad_norm, global_step)

        writer.add_scalar(
            "losses/sum_total_episodic_return",
            np.sum(total_episodic_return),
            global_step,
        )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("eval/travel_time", np.mean(env.travel_time), global_step)
        writer.add_scalar(
            "eval/social_welfare", np.mean(env.time_cost_burden), global_step
        )
        writer.add_scalar("eval/combined_cost", np.mean(env.combined_cost), global_step)
        writer.add_scalar("road/road_0_profit", env.road_profits[0], global_step)
        writer.add_scalar("road/road_1_profit", env.road_profits[1], global_step)
        writer.add_scalar(
            "road/profit_delta", env.road_profits[0] - env.road_profits[1], global_step
        )
        writer.add_scalar(
            "road/road_0_price_range", env.agent_price_range[0], global_step
        )
        writer.add_scalar(
            "road/road_1_price_range", env.agent_price_range[1], global_step
        )
        # writer.add_scalar("road/road_0_action_entropy", agent_entropy[0], global_step)
        # writer.add_scalar("road/road_1_action_entropy", agent_entropy[1], global_step)

        # writer.add_scalar("summary:travel_time", np.mean(env.travel_time), global_step)

    # pbar.set_description(
    #     "Episode:"
    #     + str(episode)
    #     + ", Combined Cost Score: "
    #     + str(np.mean(env.combined_cost))
    # )

    agent.eval()

    # wandb.run.summary["travel_time"] = np.mean(env.travel_time)
    # exit()
    if not args.track:
        exit()

    with torch.no_grad():
        if args.random_cars:
            print('evaluating')
            # trained agent performance
            trained_agent_means_tt = []
            trained_agent_means_sc = []
            trained_agent_means_cc = []
            trained_agent_means_pr = []
            # random agent performance
            random_agent_means_tt = []
            random_agent_means_sc = []
            random_agent_means_cc = []
            random_agent_means_pr = []
            for eval_n_cars in [500, 600, 650, 700, 750, 800, 850, 900, 1000]:
                print(eval_n_cars)
                for episode in range(50):
                    # trained agent on seed/n_cars
                    obs, infos = env.reset(seed=None, set_np_seed=episode, random_cars=args.random_cars)
                    obs = batchify_obs(obs, device)
                    while not any(terms) and not any(truncs):
                        if args.action_masks:
                            s_masks = torch.tensor([info[agt]['action_mask'] for agt in env.agents])
                            actions, logprobs, _, values = agent.get_action_and_value(obs, action_masks=s_masks)
                        else:
                            actions, logprobs, _, values = agent.get_action_and_value(obs)
                        obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                        obs = batchify_obs(obs, device)
                        terms = [terms[a] for a in terms]
                        truncs = [truncs[a] for a in truncs]
                    trained_agent_means_tt.append(np.mean(env.travel_time))
                    trained_agent_means_sc.append(np.mean(env.time_cost_burden))
                    trained_agent_means_cc.append(np.mean(env.combined_cost))
                    trained_agent_means_pr.append(env.road_profits[0] + env.road_profits[1])

                    # run random agent on the env
                    obs, infos = env.reset(seed=None, set_np_seed=episode, random_cars=args.random_cars)
                    # obs = batchify_obs(obs, device)
                    terms = [False]
                    truncs = [False]
                    while env.agents:
                        # this is where you would insert your policy
                        actions = {agent: randint(0, 2) for agent in env.agents}
                        observations, rewards, terminations, truncations, infos = env.step(actions)
                    random_agent_means_tt.append(np.mean(env.travel_time))
                    random_agent_means_sc.append(np.mean(env.time_cost_burden))
                    random_agent_means_cc.append(np.mean(env.combined_cost))
                    random_agent_means_pr.append(env.road_profits[0] + env.road_profits[1])
                if args.track:
                    wandb.run.summary[f"{eval_n_cars}/travel_time"] = np.mean(trained_agent_means_tt)
                    wandb.run.summary[f"{eval_n_cars}/social_cost"] = np.mean(trained_agent_means_sc)
                    wandb.run.summary[f"{eval_n_cars}/combined_cost"] = np.mean(trained_agent_means_cc)
                    wandb.run.summary[f"{eval_n_cars}/profit"] = np.mean(trained_agent_means_pr)
                    wandb.run.summary[f"{eval_n_cars}/gini_coef_tt"] = ineq.gini(trained_agent_means_tt)
                    wandb.run.summary[f"{eval_n_cars}/atki_indx_tt"] = ineq.atkinson.index(trained_agent_means_tt, epsilon=0.5)

                    wandb.run.summary[f"{eval_n_cars}/rng_travel_time"] = np.mean(random_agent_means_tt)
                    wandb.run.summary[f"{eval_n_cars}/rng_social_cost"] = np.mean(random_agent_means_sc)
                    wandb.run.summary[f"{eval_n_cars}/rng_combined_cost"] = np.mean(random_agent_means_cc)
                    wandb.run.summary[f"{eval_n_cars}/rng_profit"] = np.mean(random_agent_means_pr)

                    # Gap to Random Agent
                    wandb.run.summary[f"{eval_n_cars}/tt_performance_gap"] = np.mean(trained_agent_means_tt) - np.mean(
                        random_agent_means_tt)
                    wandb.run.summary[f"{eval_n_cars}/sc_performance_gap"] = np.mean(trained_agent_means_sc) - np.mean(
                        random_agent_means_sc)
                    wandb.run.summary[f"{eval_n_cars}/cc_performance_gap"] = np.mean(trained_agent_means_cc) - np.mean(
                        random_agent_means_cc)
                    wandb.run.summary[f"{eval_n_cars}/gini_perf_gap"] = ineq.gini(trained_agent_means_tt) - ineq.gini(
                        random_agent_means_tt)
                    wandb.run.summary[f"{eval_n_cars}/atki_indx_tt"] = ineq.atkinson.index(trained_agent_means_tt,
                                                                            epsilon=0.5) - ineq.atkinson.index(
                        random_agent_means_tt, epsilon=0.5)

        else:
            trained_agent_means_tt = []
            trained_agent_means_sc = []
            trained_agent_means_cc = []
            trained_agent_means_pr = []
            for episode in range(50):
                obs, infos = env.reset(seed=None, set_np_seed=episode, random_cars=args.random_cars)
                obs = batchify_obs(obs, device)
                terms = [False]
                truncs = [False]
                while not any(terms) and not any(truncs):
                    if args.action_masks:
                        s_masks = torch.tensor([info[agt]['action_mask'] for agt in env.agents])
                        actions, logprobs, _, values = agent.get_action_and_value(obs, action_masks=s_masks)
                    else:
                        actions, logprobs, _, values = agent.get_action_and_value(obs)
                    obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                    obs = batchify_obs(obs, device)
                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]

                print("VOT/TIMESeed:", episode)
                # print(env.car_dist_arrival)
                # print(env.car_vot_arrival)
                print(f"Travel Time: {np.mean(env.travel_time)}")
                print(f"Time-Cost Burden Score: {np.mean(env.time_cost_burden)}")
                print(f"Combined Cost Score: {np.mean(env.combined_cost)}")
                trained_agent_means_tt.append(np.mean(env.travel_time))
                trained_agent_means_sc.append(np.mean(env.time_cost_burden))
                trained_agent_means_cc.append(np.mean(env.combined_cost))
                trained_agent_means_pr.append(env.road_profits[0] + env.road_profits[1])

            random_agent_means_tt = []
            random_agent_means_sc = []
            random_agent_means_cc = []
            for episode in range(50):
                obs, infos = env.reset(seed=None, set_np_seed=episode, random_cars=args.random_cars)
                # obs = batchify_obs(obs, device)
                terms = [False]
                truncs = [False]
                while env.agents:
                    # this is where you would insert your policy
                    actions = {agent: randint(0, 2) for agent in env.agents}
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                print("VOT/TIMESeed:", episode)
                print(f"Travel Time: {np.mean(env.travel_time)}")
                print(f"Time-Cost Burden Score: {np.mean(env.time_cost_burden)}")
                print(f"Combined Cost Score: {np.mean(env.combined_cost)}")
                random_agent_means_tt.append(np.mean(env.travel_time))
                random_agent_means_sc.append(np.mean(env.time_cost_burden))
                random_agent_means_cc.append(np.mean(env.combined_cost))
                random_agent_means_cc.append(env.road_profits[0] + env.road_profits[1])


        if args.track:
            # Pure values
            wandb.run.summary["travel_time"] = np.mean(trained_agent_means_tt)
            wandb.run.summary["social_cost"] = np.mean(trained_agent_means_sc)
            wandb.run.summary["combined_cost"] = np.mean(trained_agent_means_cc)
            wandb.run.summary["profit"] = np.mean(trained_agent_means_pr)
            wandb.run.summary['gini_coef_tt'] = ineq.gini(trained_agent_means_tt)
            wandb.run.summary['atki_indx_tt'] = ineq.atkinson.index(trained_agent_means_tt, epsilon=0.5)

            # Gap to Random Agent
            wandb.run.summary["tt_performance_gap"] = np.mean(trained_agent_means_tt) - np.mean(random_agent_means_tt)
            wandb.run.summary["sc_performance_gap"] = np.mean(trained_agent_means_sc) - np.mean(random_agent_means_sc)
            wandb.run.summary["cc_performance_gap"] = np.mean(trained_agent_means_cc) - np.mean(random_agent_means_cc)
            wandb.run.summary['gini_perf_gap'] = ineq.gini(trained_agent_means_tt) - ineq.gini(random_agent_means_tt)
            wandb.run.summary['atki_indx_tt'] = ineq.atkinson.index(trained_agent_means_tt, epsilon=0.5) - ineq.atkinson.index(random_agent_means_tt, epsilon=0.5)

        print(
            "trained agents:",
            (
                nmin(trained_agent_means_tt),
                quantile(trained_agent_means_tt, 0.25),
                mean(trained_agent_means_tt),
                median(trained_agent_means_tt),
                quantile(trained_agent_means_tt, 0.75),
                nmax(trained_agent_means_tt),
                std(trained_agent_means_tt),
                ineq.gini(trained_agent_means_tt),
                ineq.atkinson.index(trained_agent_means_tt, epsilon=0.5),
            ),
            "\n",
            (
                nmin(trained_agent_means_sc),
                quantile(trained_agent_means_sc, 0.25),
                mean(trained_agent_means_sc),
                median(trained_agent_means_sc),
                quantile(trained_agent_means_sc, 0.75),
                nmax(trained_agent_means_sc),
                std(trained_agent_means_sc),
                ineq.gini(trained_agent_means_sc),
                ineq.atkinson.index(trained_agent_means_sc, epsilon=0.5),
            ),
            "\n",
            (
                nmin(trained_agent_means_cc),
                quantile(trained_agent_means_cc, 0.25),
                mean(trained_agent_means_cc),
                median(trained_agent_means_cc),
                quantile(trained_agent_means_cc, 0.75),
                nmax(trained_agent_means_cc),
                std(trained_agent_means_cc),
                ineq.gini(trained_agent_means_cc),
                ineq.atkinson.index(trained_agent_means_cc, epsilon=0.5),
            ),
        )
        print(
            "random agents:",
            (
                nmin(random_agent_means_tt),
                quantile(random_agent_means_tt, 0.25),
                mean(random_agent_means_tt),
                median(random_agent_means_tt),
                quantile(random_agent_means_tt, 0.75),
                nmax(random_agent_means_tt),
                std(random_agent_means_tt),
                ineq.gini(random_agent_means_tt),
                ineq.atkinson.index(random_agent_means_tt, epsilon=0.5),
            ),
            "\n",
            (
                nmin(random_agent_means_sc),
                quantile(random_agent_means_sc, 0.25),
                mean(random_agent_means_sc),
                median(random_agent_means_sc),
                quantile(random_agent_means_sc, 0.75),
                nmax(random_agent_means_sc),
                std(random_agent_means_sc),
                ineq.gini(random_agent_means_sc),
                ineq.atkinson.index(random_agent_means_sc, epsilon=0.5),
            ),
            "\n",
            (
                nmin(random_agent_means_cc),
                quantile(random_agent_means_cc, 0.25),
                mean(random_agent_means_cc),
                median(random_agent_means_cc),
                quantile(random_agent_means_cc, 0.75),
                nmax(random_agent_means_cc),
                std(random_agent_means_cc),
                ineq.gini(random_agent_means_cc),
                ineq.atkinson.index(random_agent_means_cc, epsilon=0.5),
            ),
        )
