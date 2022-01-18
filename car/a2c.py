import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np
import car.gcolab as gcolab
import importlib
import matplotlib.animation as animation


class A2C:
    def __init__(self, net, optim, gamma=0.90, policy_weight=1., value_weight=0.01, entropy_weight=5., entropy_range=(0.9, 1.1), clip_grad=0, tensor_board=False, checkpoint=None, device=None) -> None:
        self.net = net
        self.optim = optim
        self.gamma = gamma
        self.sum_rewards = []

        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.entropy_range = entropy_range

        self.clip_grad = clip_grad

        self.writer = None
        if tensor_board:
            log_dir = 'runs/a2c'
            self.writer = SummaryWriter(log_dir)

            delete_files(log_dir)

        delete_files("images")

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda':
            print("Using CUDA")

        self.device = device
        self.net.to(self.device)

        self.time_start = time.time()

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def optimize_model(self):

        rewards = torch.tensor(
            self.rewards, dtype=torch.float32, device=self.device)

        active_states = ~torch.tensor(self.done_states, device=self.device)
        not_dones = active_states[1:]

        truncateds = torch.tensor(self.truncateds, device=self.device)

        td_targets = torch.zeros(
            (self.n_steps, self.num_env), dtype=torch.float32, device=self.device)

        Svalues = torch.stack(self.Svalues)

        old = Svalues[-1].detach()

        for i in reversed(range(self.n_steps)):
            old = rewards[i] + self.gamma * \
                (old * not_dones[i] +
                 Svalues[i+1].detach() * truncateds[i])
            td_targets[i] = old

        # td_targets = (td_targets - td_targets.mean(axis=0)) / \
        #     (td_targets.std(axis=0) + 1e-8)

        advantage = (td_targets - Svalues[:-1]) * active_states[:-1]
        value_loss = advantage.pow(2).mul(0.5).mean()

        log_probs = torch.stack(self.log_probs)
        policy_loss = -(advantage.detach() * log_probs).mean()

        entropy_loss = -torch.stack(self.entropies).mean()

        loss = self.policy_weight * policy_loss + self.entropy_weight * \
            entropy_loss + self.value_weight * value_loss

        self.optim.zero_grad()
        loss.backward()

        n = self.step / (self.num_env * self.n_steps)
        if n % 200 == 0:
            grads = []
            for param in self.net.parameters():
                grads.append(round(param.grad.norm().item(), 2))
            print("    norms of grads = ", grads)

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           self.clip_grad)

        if n % 200 == 0:
            print(
                f"    policy={round(policy_loss.item(),3)}, value={round(value_loss.item(),2)}, entropy={round(entropy_loss.item(),2)}, en_w={round(self.entropy_weight,2)}")

            if gcolab.save:
                self.save_checkpoint("check")

        if n % 20 == 0:
            importlib.reload(gcolab)
            self.entropy_range = gcolab.entropy_range
            self.value_weight = gcolab.value_weight

            if -entropy_loss.item() < self.entropy_range[0]:
                self.entropy_weight += 0.2
            elif -entropy_loss.item() > self.entropy_range[1]:
                self.entropy_weight = max(0.01, self.entropy_weight - 0.2)

        self.optim.step()

        if self.writer and n % 100 == 0:
            self.writer.add_scalar(
                "loss/policy", policy_loss.item(), self.step)
            self.writer.add_scalar(
                "loss/value", value_loss.item(), self.step)
            self.writer.add_scalar(
                "loss/entropy", entropy_loss.item(), self.step)

    def decide(self, states):

        #         /content/a2c.py:110: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  ../torch/csrc/utils/tensor_new.cpp:201.)
        states = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=self.device)[:, None]

        # b_* is batch_*
        b_action_logits, b_Svalue = self.net(states)

        dist = torch.distributions.Categorical(logits=b_action_logits)
        b_action = dist.sample()
        b_log_prob = dist.log_prob(b_action)
        b_entropy = dist.entropy()

        self.log_probs.append(b_log_prob)
        self.entropies.append(b_entropy)
        self.Svalues.append(b_Svalue.squeeze())

        return b_action.tolist()

    def plot_rewards(self):
        _, ax = plt.subplots(figsize=(7, 7))
        ax.plot(self.sum_rewards)

    def train(self, env, ev_env, max_steps=1e7, n_steps=5):

        self.n_steps = n_steps
        self.num_env = env.num_env
        self.ev_env = ev_env

        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.Svalues = []
        self.done_states = [[False]*env.num_env]  # False = not in done
        self.truncateds = []

        env.gamma = self.gamma

        self.step = 0
        k = 0
        states = env.reset()

        self.time_start = time.time()

        while self.step < max_steps and k < 4:

            self.net.train()

            for _ in range(n_steps):

                actions = self.decide(states)
                states, rewards, dones, truncateds = env.step(actions)

                self.rewards.append(rewards)
                self.done_states.append(dones)
                self.truncateds.append(truncateds)

            _, b_Svalue = self.net(torch.as_tensor(
                np.array(states), dtype=torch.float32, device=self.device)[:, None])

            self.Svalues.append(b_Svalue.squeeze())

            self.optimize_model()

            self.log_probs = self.log_probs[:0]
            self.rewards = self.rewards[:0]
            self.entropies = self.entropies[:0]
            self.Svalues = self.Svalues[:0]
            self.done_states = [self.done_states[-1]]
            self.truncateds = self.truncateds[:0]

            n = self.step / (self.num_env * self.n_steps)

            self.step += env.num_env * n_steps

            if n % 200 == 0:

                k += 1

                mean_steps = self.evaluate()

                if self.writer:
                    self.writer.add_scalar(
                        "Goal/score", mean_steps, self.step)
        if self.writer:
            self.writer.close()

    def evaluate(self, num_episodes=5, ev_env=None, record=False):

        if ev_env:
            self.ev_env = ev_env
        self.ev_env.gamma = self.gamma

        finished = 0
        # sum_rewards = []
        num_steps = []
        how_fast = []

        time_ev = round((time.time() - self.time_start)/60, 1)

        self.net.eval()

        for ep in range(num_episodes):

            # sum_rewards.append(0)
            avg_velosity = 0
            way = []

            state = self.ev_env.reset()

            if record:
                self.records = [(self.ev_env.area_trajectory(), state, 0)]

            for step in range(1, int(1e7)):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)[
                    None, None]

                with torch.no_grad():
                    action_logits, _ = self.net(state)

                action = torch.argmax(action_logits)

                state, reward, done, info = self.ev_env.step(action)

                # sum_rewards[-1] += reward
                way.append(round(reward, 2))

                avg_velosity += (info["velosity"] - avg_velosity) / step

                if record:
                    self.records.append(
                        (self.ev_env.area_trajectory(), state, info["velosity"]))

                if done:
                    break

            num_steps.append(step)
            # насколько быстрее, чем если бы двигался 1 пиксель за ход
            pct = 0 if not info["finished"] else (
                round(len(self.ev_env.tr.path)/step, 2))

            how_fast.append((round(avg_velosity, 1), pct))

            lway = -min(7, len(way))

            print(info["reason"], "==>", info["truncated"], way[lway:])

            finished += info["finished"]

            # self.ev_env.save_area_to_image(f"{time_ev}min_{step}")

            # print(rewards)
            if self.writer:
                self.writer.add_image(f"{self.step}",
                                      self.ev_env.area_trajectory(), self.step, dataformats='HW')

        print(
            f"TIME= {time_ev} min., FAST (avg_velosity, ratio_steps)={how_fast},\nSTEPS= {num_steps}, finished {finished} / {num_episodes}\n\n")

        return sum(num_steps)/num_episodes

    def animate(self, env):
        self.evaluate(num_episodes=1, ev_env=env, record=True)
        fig, axes = plt.subplots(1, 2, figsize=(7, 7))
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)

        ims = []
        for i, r in enumerate(self.records):
            im = axes[0].imshow(r[0], animated=True)
            im1 = axes[1].imshow(r[1], animated=True)
            im2 = axes[0].text(30, 35, f'{r[2]:.1f} km/h',
                               fontsize=25, color="white", animated=True)
            ims.append([im, im1, im2])
            if i == 0:
                axes[0].imshow(r[0])
                axes[1].imshow(r[1])
        ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True, repeat_delay=1000)
        return ani

    def save_checkpoint(self, name):
        name = f'{name}_point.pth'
        print("Saving checkpoint", name)
        checkpoint = {'step': self.step,
                      "net": str(self.net),
                      'model_state_dict': self.net.state_dict(),
                      'optimizer_state_dict': self.optim.state_dict()}
        if os.path.exists(name):
            os.remove(name)
        torch.save(checkpoint, name)
        return

    def load_checkpoint(self, name):
        print("loading checkpoint", name)
        checkpoint = torch.load(name, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
        return


def delete_files(dir_):
    if not os.path.exists(dir_):
        return
    for root, dirs, files in os.walk(dir_):
        for file in files:
            os.remove(os.path.join(root, file))
