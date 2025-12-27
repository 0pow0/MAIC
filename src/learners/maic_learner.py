import copy
import os
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mve import MVEValueNet
import torch as th
from torch.optim import RMSprop


class MAICLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.msg_params = None

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mve_enabled = getattr(args, "mve_enabled", False)
        self.mve_pretrain_steps = getattr(args, "mve_pretrain_steps", 0)
        self.mve_train_steps = getattr(args, "mve_train_steps", 0)
        self.mve_unlearn_steps = getattr(args, "mve_unlearn_steps", 0)
        self.mve_lambda = getattr(args, "mve_lambda", 0.0)
        self.mve = None
        self.mve_optimiser = None
        self.unlearn_optimiser = None
        self._phase = "pretrain"
        if self.mve_enabled:
            msg_dim = args.n_agents * args.n_actions
            input_dim = args.rnn_hidden_dim + msg_dim
            hidden_dim = getattr(args, "mve_hidden_dim", args.rnn_hidden_dim)
            mve_lr = getattr(args, "mve_lr", args.lr)
            unlearn_lr = getattr(args, "mve_unlearn_lr", args.lr)
            self.mve = MVEValueNet(input_dim, hidden_dim)
            self.mve_optimiser = RMSprop(params=self.mve.parameters(), lr=mve_lr,
                                         alpha=args.optim_alpha, eps=args.optim_eps)
            if hasattr(self.mac.agent, "message_parameters"):
                self.msg_params = self.mac.agent.message_parameters()
            else:
                self.msg_params = self.params
            self.unlearn_optimiser = RMSprop(params=self.msg_params, lr=unlearn_lr,
                                             alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if self.mve_enabled:
            phase = self._get_phase(t_env)
            if phase != self._phase:
                self._phase = phase
                self.logger.console_logger.info("MAIC phase switched to {}".format(phase))
            if phase == "mve":
                self._train_mve(batch, t_env)
                return
            if phase == "unlearn":
                self._train_unlearn(batch, t_env)
                return
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # NOTE: record logging signal
        prepare_for_logging = True if t_env - self.log_stats_t >= self.args.learner_log_interval else False

        logs = []
        losses = []

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs, returns_ = self.mac.forward(batch, t=t, 
                prepare_for_logging=prepare_for_logging,
                train_mode=True,
                mixer=self.target_mixer,
            )
            mac_out.append(agent_outs)
            if prepare_for_logging and 'logs' in returns_:
                logs.append(returns_['logs'])
                del returns_['logs']
            losses.append(returns_)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss += external_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self._log_for_loss(loss_dict, t_env)

            self.log_stats_t = t_env

    def _get_phase(self, t_env):
        pre_end = max(self.mve_pretrain_steps, 0)
        mve_end = pre_end + max(self.mve_train_steps, 0)
        if self.mve_pretrain_steps > 0 and t_env < pre_end:
            return "pretrain"
        if self.mve_train_steps > 0 and t_env < mve_end:
            return "mve"
        if self.mve_unlearn_steps > 0:
            return "unlearn"
        return "mve" if self.mve_train_steps > 0 else "pretrain"

    def _mac_forward(self, batch: EpisodeBatch, msg_mask=None, return_mve=False):
        mac_out = []
        mve_h = []
        mve_msg = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, returns_ = self.mac.forward(
                batch,
                t=t,
                test_mode=False,
                msg_mask=msg_mask,
                return_mve=return_mve,
            )
            mac_out.append(agent_outs)
            if return_mve:
                mve_h.append(returns_["mve_h"])
                mve_msg.append(returns_["mve_msg"])
        mac_out = th.stack(mac_out, dim=1)
        if return_mve:
            return mac_out, th.stack(mve_h, dim=1), th.stack(mve_msg, dim=1)
        return mac_out

    def _compute_joint_q(self, mac_out, batch: EpisodeBatch):
        actions = batch["actions"][:, :-1]
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        else:
            chosen_action_qvals = chosen_action_qvals.sum(dim=2, keepdim=True)
        return chosen_action_qvals

    def _train_mve(self, batch: EpisodeBatch, t_env: int):
        if self.mve is None:
            return
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        with th.no_grad():
            mac_out, mve_h, mve_msg = self._mac_forward(batch, return_mve=True)
            q_real = self._compute_joint_q(mac_out, batch)
            delta_qs = []
            for agent_i in range(self.args.n_agents):
                msg_mask = th.ones(batch.batch_size, self.args.n_agents, device=batch.device)
                msg_mask[:, agent_i] = 0
                mac_out_null = self._mac_forward(batch, msg_mask=msg_mask)
                q_null = self._compute_joint_q(mac_out_null, batch)
                delta_qs.append(q_real - q_null)
            delta_q = th.stack(delta_qs, dim=2)

        bs, max_t, n_agents, _ = mve_h.shape
        msg_flat = mve_msg[:, :-1].reshape(bs, max_t - 1, n_agents, -1)
        h_flat = mve_h[:, :-1]
        inputs = th.cat([h_flat, msg_flat], dim=-1).reshape(-1, h_flat.shape[-1] + msg_flat.shape[-1])
        preds = self.mve(inputs).view(bs, max_t - 1, n_agents, 1)

        mask_expanded = mask.unsqueeze(2).unsqueeze(3).expand_as(preds)
        mse = (preds - delta_q.detach()) ** 2
        loss = (mse * mask_expanded).sum() / mask_expanded.sum()

        self.mve_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.mve.parameters(), self.args.grad_norm_clip)
        self.mve_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("mve_loss", loss.item(), t_env)
            self.logger.log_stat("mve_grad_norm", grad_norm, t_env)
            self.logger.log_stat("mve_delta_q_mean", delta_q.mean().item(), t_env)
            self.log_stats_t = t_env

    def _train_unlearn(self, batch: EpisodeBatch, t_env: int):
        if self.mve is None or self.unlearn_optimiser is None:
            return
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        _, mve_h, mve_msg = self._mac_forward(batch, return_mve=True)

        bs, max_t, n_agents, _ = mve_h.shape
        msg_flat = mve_msg[:, :-1].reshape(bs, max_t - 1, n_agents, -1)
        h_flat = mve_h[:, :-1]
        inputs = th.cat([h_flat, msg_flat], dim=-1).reshape(-1, h_flat.shape[-1] + msg_flat.shape[-1])
        with th.no_grad():
            v_hat = self.mve(inputs).view(bs, max_t - 1, n_agents, 1)

        msg_l1 = msg_flat.abs().sum(-1, keepdim=True)
        advantage = (v_hat - self.mve_lambda * msg_l1).detach()
        mask_expanded = mask.unsqueeze(2).unsqueeze(3).expand_as(msg_l1)
        unlearn_loss = -(advantage * msg_l1 * mask_expanded).sum() / mask_expanded.sum()

        self.unlearn_optimiser.zero_grad()
        unlearn_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.msg_params, self.args.grad_norm_clip)
        self.unlearn_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("unlearn_loss", unlearn_loss.item(), t_env)
            self.logger.log_stat("unlearn_grad_norm", grad_norm, t_env)
            self.logger.log_stat("unlearn_adv_mean", advantage.mean().item(), t_env)
            self.logger.log_stat("unlearn_msg_l1_mean", msg_l1.mean().item(), t_env)
            self.log_stats_t = t_env

    def _process_loss(self, losses: list, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            for k, v in item.items():
                if str(k).endswith('loss'):
                    loss_dict[k] = loss_dict.get(k, 0) + v
                    total_loss += v
        for k in loss_dict.keys():
            loss_dict[k] /= batch.max_seq_length
        total_loss /= batch.max_seq_length
        return total_loss, loss_dict

    def _log_for_loss(self, losses: dict, t):
        for k, v in losses.items():
            self.logger.log_stat(k, v.item(), t)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.mve is not None:
            self.mve.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.mve is not None:
            th.save(self.mve.state_dict(), "{}/mve.th".format(path))
            th.save(self.mve_optimiser.state_dict(), "{}/mve_opt.th".format(path))
            th.save(self.unlearn_optimiser.state_dict(), "{}/unlearn_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.mve is not None:
            mve_path = os.path.join(path, "mve.th")
            mve_opt_path = os.path.join(path, "mve_opt.th")
            unlearn_opt_path = os.path.join(path, "unlearn_opt.th")
            if os.path.exists(mve_path):
                self.mve.load_state_dict(th.load(mve_path, map_location=lambda storage, loc: storage))
            if os.path.exists(mve_opt_path):
                self.mve_optimiser.load_state_dict(th.load(mve_opt_path, map_location=lambda storage, loc: storage))
            if os.path.exists(unlearn_opt_path):
                self.unlearn_optimiser.load_state_dict(th.load(unlearn_opt_path,
                                                               map_location=lambda storage, loc: storage))
