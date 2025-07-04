import argparse
import os
import time

import numpy as np
from numpy import min as nmin
from numpy import max as nmax
from numpy import mean, quantile, median, std

from str2bool import str2bool as strtobool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from tqdm import tqdm, trange
import inequalipy as ineq

from BigPZEnv import TNTPParallelEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Run Settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    # Environment Parameters
    parser.add_argument("--timesteps", type=int, default=1800,
                        help="total episodes of the experiments")
    parser.add_argument("--k_depth", type=int, default=2,
                        help="depth of routes offered")
    parser.add_argument("--random_init_road_cost", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="fix initial road costs or randomize them")
    parser.add_argument("--lambd", type=float, default=0.9,
                        help="lambda paramater for QRE")
    # parser.add_argument("--tntp_path", type=str, default="/Users/behradkoohy/Development/TransportationNetworks/SiouxFalls/SiouxFalls",
    #                     help="Path to TNTP files")
    parser.add_argument("--tntp_path", type=str,
                        default="/Users/behradkoohy/Development/TransportationNetworks/Small-Seq-Example/Small-Seq-Example",
                        help="Path to TNTP files")

    # Algorithm Parameters
    parser.add_argument("--num-episodes", type=int, default=10000,
                        help="total episodes of the experiments")
    parser.add_argument("--num-steps", type=int, default=5500,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--eps_per_update", type=int, default=8,
                        help="Number of episodes per update. If 1, same behaviour as before.")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
                        help="the K epochs to update the policy")
    parser.add_argument("--action-masks", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether invalid actions are masked or not.")

    # Algorithm Hyperparameters
    parser.add_argument("--reward-clip", type=float, default=0,
                        help="The value to clip the reward. If left at 0, the reward will not be clipped")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
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

    args = parser.parse_args()
    args.batch_size = 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # args.num_steps = ((args.num_steps + 1) * 5)
    args.num_steps = ((args.num_steps + 1) * args.eps_per_update)
    args.total_timesteps = args.timesteps * args.num_episodes
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
    def __init__(self, obs_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear((obs_size), 512)),
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
            layer_init(nn.Linear((obs_size), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 3), std=0.01),
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        # return self.critic(self.network(x / 255.0))
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_masks=None):
        x = x.float()
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        if not args.action_masks:
            return action, probs.log_prob(action), probs.entropy(), self.critic(self.network(x) if args.network_size == 5 else x)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("----------- RUN DETAILS -----------")
    print(f"Experiment Name: {args.exp_name}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Num Episodes: {args.num_episodes}")
    print(f"TNTP Path: {args.tntp_path}")
    print("-----------      END     -----------")
    run_name = f"MMRP_Online__5EpisodesPerRollOut:MinimiseTT__{args.total_timesteps}__{args.k_depth}__{int(time.time())}"
    n_timesteps = args.timesteps
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

    num_actions = 3
    env = TNTPParallelEnv(
        timesteps=args.timesteps,
        k_depth=args.k_depth,
        tntp_path=args.tntp_path,
        random_initial_road_cost=args.random_init_road_cost,
        lambd=args.lambd
    )
    num_agents = len(env.possible_agents)
    obs_size = env.observation_space(env.possible_agents[0]).shape
    ind_obs_size = 9
    """ LEARNER SETUP """
    agent = Agent(ind_obs_size).to(device)
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
    rb_obs = torch.zeros((args.total_timesteps, num_agents) + (ind_obs_size,),
                         dtype=torch.float32).to(
        device
    )
    rb_actions = torch.zeros((args.total_timesteps, num_agents), dtype=torch.float32).to(
        device
    )
    rb_logprobs = torch.zeros((args.total_timesteps, num_agents), dtype=torch.float32).to(
        device
    )
    rb_rewards = torch.zeros((args.total_timesteps + 1, num_agents), dtype=torch.float32).to(
        device
    )
    rb_terms = torch.zeros((args.total_timesteps + 1, num_agents), dtype=torch.float32).to(device)
    rb_values = torch.zeros((args.total_timesteps + 1, num_agents), dtype=torch.float32).to(
        device
    )
    rb_action_masks = torch.zeros((args.total_timesteps, num_agents, num_actions),
                                  dtype=torch.float32).to(
        device
    )

    num_updates = args.total_timesteps // args.batch_size
    start_time = time.time()

    completed_eps = 0
    pbar = tqdm(range(args.num_episodes), position=0, leave=False, colour='green')
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

        for episode in trange(args.eps_per_update, position=1, leave=False, colour='red'):
            with torch.no_grad():
                next_obs, info = env.reset()
                total_episodic_return = 0
                while env.agents:
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

        # print(
        #     f"At Episode {completed_eps}: Batch episodic return: {np.mean(batch_episodic_return)}, BER/Car: {np.mean(batch_episodic_return) / env.n_cars}")
        # # print(
        # #     f"At Episode {completed_eps}: Batch episodic difference: {np.mean(batch_episodic_difference)}, BED/Car: {np.mean(batch_episodic_difference) / env.n_cars}")
        # print(f"Batch n_cars: {batch_n_cars}")
        # TODO: we need to find a way of fixing how long an episode takes because we are running out of space in the buffer
        # TODO: this might be because we aren't calculating the buffer size correctly.
        # I think this has been fixed?
        pbar.update(args.eps_per_update)
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
        b_obs = torch.flatten(rb_obs[:batch_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:batch_step + 1], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:batch_step + 1], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:batch_step + 1], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:batch_step + 1], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:batch_step + 1], start_dim=0, end_dim=1)
        if args.action_masks:
            b_action_masks = torch.flatten(rb_action_masks[:batch_step + 1], start_dim=0, end_dim=1)
        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(args.update_epochs):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
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

                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()

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
        if args.track:
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # for agt_tr in range(num_agents):
            #     writer.add_scalar(f"losses/agent_{agt_tr}_advantage", rb_advantages[:, agt_tr].mean().item(), global_step)
            #     writer.add_scalar(f"losses/agent_{agt_tr}_logprobs", rb_logprobs[:, agt_tr].max().item(), global_step)
            #     writer.add_scalar(f"losses/agent_{agt_tr}_returns_means", rb_returns[:, agt_tr].mean().item(), global_step)
            #     writer.add_scalar(f"losses/agent_{agt_tr}_returns_vars", rb_returns[:, agt_tr].var().item(), global_step)
            #     # writer.add_scalar(f"road/road_{agt_tr}_profit", env.road_profits[agt_tr], global_step)
            #     # writer.add_scalar(f"road/road_{agt_tr}_price_range", env.agent_price_range[agt_tr], global_step)
            #     # writer.add_scalar(f"road/road_{agt_tr}_max_price", env.agent_maxes[agt_tr], global_step)
            #     # writer.add_scalar(f"road/road_{agt_tr}_min_price", env.agent_mins[agt_tr], global_step)
            #     # writer.add_scalar(f"road/road_{agt_tr}_med_price", np.median(env.agent_prices[agt_tr]), global_step)
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
    with torch.no_grad():
        print('evaluating')
        trained_agent_means_tt = []
        trained_agent_means_sc = []
        trained_agent_means_cc = []
        trained_agent_means_pr = []
        for episode in trange(50):
            obs, infos = env.reset(seed=episode)
            obs = batchify_obs(obs, device)
            while env.agents:
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
        if args.track:
            wandb.run.summary["travel_time"] = np.mean(trained_agent_means_tt)
            wandb.run.summary["social_cost"] = np.mean(trained_agent_means_sc)
            wandb.run.summary["combined_cost"] = np.mean(trained_agent_means_cc)
            wandb.run.summary["profit"] = np.mean(trained_agent_means_pr)
            wandb.run.summary["gini_coef_tt"] = ineq.gini(trained_agent_means_tt)
            wandb.run.summary["atki_indx_tt"] = ineq.atkinson.index(trained_agent_means_tt,
                                                                                   epsilon=0.5)
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