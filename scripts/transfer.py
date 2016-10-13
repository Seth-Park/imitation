import argparse
import json
import h5py
import numpy as np

from environments import rlgymenv
import gym

import policyopt
from policyopt import SimConfig, rl, util, nn, tqdm

#SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 64}, {"type": "nonlin", "func": "lrelu"}, {"type": "fc", "n": 64}, {"type": "nonlin", "func": "lrelu"}]'
SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", " func": "tanh"}]'

def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    # MDP options
    parser.add_argument('policy', type=str)
    parser.add_argument('--discount', type=float, default=.995)
    parser.add_argument('--lam', type=float, default=.97)
    parser.add_argument('--max_traj_len', type=int, default=None)
    # Optimizer
    parser.add_argument('--max_iter', type=int, default=1000000)
    parser.add_argument('--policy_max_kl', type=float, default=.01)
    parser.add_argument('--policy_cg_damping', type=float, default=.1)
    parser.add_argument('--vf_max_kl', type=float, default=.01)
    parser.add_argument('--vf_cg_damping', type=float, default=.1)
    # Sampling
    parser.add_argument('--sim_batch_size', type=int, default=None)
    parser.add_argument('--min_total_sa', type=int, default=100000)
    # Saving stuff
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--log', type=str, required=False)

    args = parser.parse_args()

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(args.policy)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[policy_key]
        import pprint
        pprint.pprint(dict(dset.attrs))

    args.policy_hidden_spec = train_args['policy_hidden_spec']
    args.env_name = train_args['env_name']

    # Initialize the MDP
    print 'Loading environment', args.env_name
    mdp = rlgymenv.RLGymMDP(args.env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    if args.max_traj_len is None:
        args.max_traj_len = mdp.env_spec.timestep_limit
    util.header('Max traj len is {}'.format(args.max_traj_len))

    # Initialize the policy and load its parameters
    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
    args.enable_obsnorm = int(enable_obsnorm)

    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    print(argstr)

    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=args.policy_hidden_spec,
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=args.policy_hidden_spec,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')
    policy.load_h5(policy_file, policy_key)

    util.header('Policy architecture')
    policy.print_trainable_variables()

    vf = rl.ValueFunc(
        hidden_spec=args.policy_hidden_spec,
        obsfeat_space=mdp.obs_space,
        enable_obsnorm=bool(args.enable_obsnorm),
        enable_vnorm=True,
        max_kl=args.vf_max_kl,
        damping=args.vf_cg_damping,
        time_scale=1./mdp.env_spec.timestep_limit,
        varscope_name='ValueFunc')

    opt = rl.SamplingPolicyOptimizer(
        mdp=mdp,
        discount=args.discount,
        lam=args.lam,
        policy=policy,
        sim_cfg=SimConfig(
            min_num_trajs=-1,
            min_total_sa=args.min_total_sa,
            batch_size=args.sim_batch_size,
            max_traj_len=args.max_traj_len),
        step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping),
        value_func=vf,
        obsfeat_fn=lambda obs: obs,
    )

    log = nn.TrainingLog(args.log, [('args', argstr)])

    for i in xrange(args.max_iter):
        iter_info = opt.step()
        log.write(iter_info, print_header=i % 20 == 0)
        if args.save_freq != 0 and i % args.save_freq == 0 and args.log is not None:
            log.write_snapshot(policy, i)

if __name__ == '__main__':
    main()
