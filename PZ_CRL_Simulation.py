import argparse
import time
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

from ParallelPZEnv import n_timesteps, n_cars


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Run Settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    # Algorithm Run Settings
    parser.add_argument("--num-episodes", type=int, default=10000,
        help="total episodes of the experiments")
    # parser.add_argument("--num-steps", type=int, default=((n_timesteps+3)*5),
    parser.add_argument("--num-steps", type=int, default=5500,
    help = "the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")

    # Args made by me
    # Agent params
    parser.add_argument("--reward-clip", type=float, default=0.1,
        help="The value to clip the reward. If left at 0, the reward will not be clipped")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    # Env params
    # parser.add_argument("--fixed-road-cost", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Sets the initial road cost to be fixed. If not set, road cost will be random at each episode.")
    # parser.add_argument("--linear-arrival-dist", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Sets the arrival dist to be linear. If not set, arrival dist will be beta dist.")
    # parser.add_argument("--normalised_observations", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Normalises the agent's observations. If not set, observations will be raw values.")
    parser.add_argument("--rewardfn", type=str, default="MaxProfit", nargs="?", const=True,
        help="reward function for agent to use. has to be predefined.")


    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.1,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")

    # Not Used
    # parser.add_argument("--num-steps", type=int, default=1,
    #     help="the number of complete episodes to run in each environment per policy rollout")
    # parser.add_argument("--num-envs", type=int, default=2,
    #     help="the number of parallel game environments")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--target-kl", type=float, default=None,
    #     help="the target KL divergence threshold")
    # parser.add_argument("--total-timesteps", type=int, default=20000000,
    #     help="total timesteps of the experiments")
    # parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to capture videos of the agent performances (check out `videos` folder)")
    # parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--seed", type=int, default=1,
    #     help="seed of the experiment")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, cuda will be enabled by default")
    args = parser.parse_args()
    args.num_envs = 2
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_steps = ((args.num_steps + 1) * 5)
    args.total_timesteps = n_timesteps * args.num_episodes
    # fmt: on
    return args


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self):
        pass

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
    def get_action_and_value(self, x, action=None):
        x = x.float()
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
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


if __name__ == "__main__":
    args = parse_args()
    print("----------- RUN DETAILS -----------")
    print("N CARS:", n_cars)
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
    # ent_coef = 0.01
    # vf_coef = 0.1
    # clip_coef = 0.1
    # gamma = 0.99
    # batch_size = 512
    # stack_size = 4
    # frame_size = (64, 64)
    # # max_cycles = 31*10
    # total_episodes = 3
    # total_episodes = 1

    """ ENV SETUP """
    # env = pistonball_v6.parallel_env(
    #     render_mode="rgb_array", continuous=False, max_cycles=max_cycles
    # )
    env = simulation_env(
        initial_road_cost="Fixed",
        fixed_road_cost=1.0,
        arrival_dist="Linear",
        normalised_obs=True,
        road0_capacity=15,
        road0_fftraveltime=20,
        road1_capacity=30,
        road1_fftraveltime=20,
        reward_fn=args.rewardfn
    )
    # max_cycles = env.timesteps * 10
    num_agents = 2
    num_actions = 3
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)
    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0

    rb_obs = torch.zeros((args.num_steps, num_agents) + (12,), dtype=torch.float32).to(
        device
    )
    rb_actions = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(
        device
    )
    rb_logprobs = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(
        device
    )
    rb_rewards = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(
        device
    )
    rb_terms = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(device)
    rb_values = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(
        device
    )

    num_updates = args.total_timesteps // args.batch_size
    start_time = time.time()

    """ TRAINING LOGIC """
    # train for n number of episodes
    pbar = tqdm(range(args.num_episodes))
    global_step = 0
    for episode in pbar:
        # for episode in range(total_episodes):
        if args.anneal_lr:
            frac = 1.0 - (episode - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, args.num_steps):
                global_step += 1
                # rollover the observation
                obs = batchify_obs(next_obs, device)
                # obs = next_obs
                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )
                if args.reward_clip != 0:
                    rewards = {
                        agt: min(max(r, -args.reward_clip), args.reward_clip)
                        for agt, r in rewards.items()
                    }
                # if episode % 100 == 0:
                #     print(env.time, ":: " , rewards, '\t\t\t', env.roadPrices, '\t\t\t', env.roadTravelTime)
                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break
            agent_actions = rb_actions[end_step - n_timesteps : end_step]
            agent_entropy = scipy.stats.entropy(agent_actions)

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + args.gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                # rb_advantages[t] = delta + args.gamma * args.gamma * rb_advantages[t + 1]
                # added in Generalised Advantage Estimation. Hopefully this smooths the learning process
                rb_advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * rb_terms[t + 1] * lastgaelam
                )
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        # b_inds = np.arange(args.batch_size)
        clip_fracs = []
        for repeat in range(args.update_epochs):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), args.batch_size):
                # select the indices we want to train on
                end = start + args.batch_size
                batch_index = b_index[start:end]

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
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

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
            "road/road_0_mean", env.agent_reward_norms_mean[0], global_step
        )
        writer.add_scalar(
            "road/road_0_var", env.agent_reward_norms_vars[0], global_step
        )
        writer.add_scalar(
            "road/road_1_mean", env.agent_reward_norms_mean[1], global_step
        )
        writer.add_scalar(
            "road/road_1_var", env.agent_reward_norms_vars[1], global_step
        )
        writer.add_scalar(
            "road/road_0_price_range", env.agent_price_range[0], global_step
        )
        writer.add_scalar(
            "road/road_1_price_range", env.agent_price_range[1], global_step
        )
        writer.add_scalar("road/road_0_action_entropy", agent_entropy[0], global_step)
        writer.add_scalar("road/road_1_action_entropy", agent_entropy[1], global_step)

        # writer.add_scalar("summary:travel_time", np.mean(env.travel_time), global_step)

    pbar.set_description(
        "Episode:"
        + str(episode)
        + ", Combined Cost Score: "
        + str(np.mean(env.combined_cost))
    )
    # print("Episode:", episode, ", Total Episodic Return:", total_episodic_return, ", Sum reward:", np.sum(total_episodic_return))

    # """ RENDER THE POLICY """
    # env = simulation_env()
    # # env = color_reduction_v0(env)
    # # env = resize_v1(env, 64, 64)
    # # env = frame_stack_v1(env, stack_size=4)
    #
    # wandb.log({"summary:travel_time": np.mean(env.travel_time), "travel_time": np.mean(env.travel_time)})
    agent.eval()
    trained_agent_means_tt = []
    trained_agent_means_sc = []
    trained_agent_means_cc = []
    # wandb.run.summary["travel_time"] = np.mean(env.travel_time)
    with torch.no_grad():
        # render 5 episodes out
        for episode in range(50):

            obs, infos = env.reset(seed=None, set_np_seed=episode)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
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

    random_agent_means_tt = []
    random_agent_means_sc = []
    random_agent_means_cc = []

    for episode in range(50):

        obs, infos = env.reset(seed=None, set_np_seed=episode)
        # obs = batchify_obs(obs, device)
        terms = [False]
        truncs = [False]
        while env.agents:
            # this is where you would insert your policy
            # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            actions = {agent: randint(0, 2) for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        print("VOT/TIMESeed:", episode)
        # print(env.car_dist_arrival)
        # print(env.car_vot_arrival)
        print(f"Travel Time: {np.mean(env.travel_time)}")
        print(f"Time-Cost Burden Score: {np.mean(env.time_cost_burden)}")
        print(f"Combined Cost Score: {np.mean(env.combined_cost)}")
        random_agent_means_tt.append(np.mean(env.travel_time))
        random_agent_means_sc.append(np.mean(env.time_cost_burden))
        random_agent_means_cc.append(np.mean(env.combined_cost))

    print("\n\n\n")
    # print("trained agent means:", np.mean(trained_agent_means_tt), np.mean(trained_agent_means_sc) ,np.mean(trained_agent_means_cc))
    # print("random agent means:", np.mean(random_agent_means_tt), np.mean(random_agent_means_sc), np.mean(random_agent_means_cc))
    wandb.run.summary["travel_time"] = np.mean(trained_agent_means_tt)
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
        ),
    )
