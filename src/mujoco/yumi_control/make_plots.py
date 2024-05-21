import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ast import literal_eval


def main():
    parser = argparse.ArgumentParser(description='Draw plots from simulation data')
    parser.add_argument('-i', '--input', default='sim_data.csv', required=False, help="path to simulation data")
    parser.add_argument('-p', '--plot', action='store_true', help='plot data')
    args = parser.parse_args()

    vector_cols = [
        'q', 'dq', 'start rot', 'end rot',
        'start translation', 'end translation', 'start vel', 'end vel',
        'constraint error', 'control error', 'desired rot',
        'desired translation', 'desired vel',
    ]

    sim_data = pd.read_csv(
        args.input, sep=';', converters={i: literal_eval for i in vector_cols}
    )

    ts = sim_data['time'].to_numpy()
    qs = np.array(sim_data['q'].tolist())
    dqs = np.array(sim_data['dq'].tolist())
    start_poses = np.array(sim_data['start translation'].tolist())
    des_poses = np.array(sim_data['desired translation'].tolist())
    end_poses = np.array(sim_data['end translation'].tolist())
    start_vels = np.array(sim_data['start vel'].tolist())
    des_vels = np.array(sim_data['desired vel'].tolist())
    end_vels = np.array(sim_data['end vel'].tolist())
    const_errors = np.array(sim_data['constraint error'].tolist())
    control_errors = np.array(sim_data['control error'].tolist())

    qs_norms = np.linalg.norm(qs, axis=1)
    dqs_norms = np.linalg.norm(dqs, axis=1)
    diff_poses = des_poses - start_poses
    diff_vels = des_vels - start_vels
    const_errors_norm = np.linalg.norm(const_errors, axis=1)
    control_errors_norm = np.linalg.norm(control_errors, axis=1)
    diff_lin_vels = np.linalg.norm(diff_vels[:, :3], axis=1)
    diff_ang_vels = np.linalg.norm(diff_vels[:, 3:], axis=1)
    end_lin_vels = np.linalg.norm(end_vels[:, :3], axis=1)
    end_ang_vels = np.linalg.norm(end_vels[:, 3:], axis=1)

    t_1500ms_idx = np.squeeze(np.argwhere(ts > 1.5))[0]
    trend_const_e = np.mean(const_errors_norm[t_1500ms_idx:])
    trend_control_e = np.mean(control_errors_norm[t_1500ms_idx:])
    trend_diff_lin = np.mean(diff_lin_vels[t_1500ms_idx:])
    trend_diff_ang = np.mean(diff_ang_vels[t_1500ms_idx:])
    trend_diff_poses = np.mean(diff_poses[t_1500ms_idx:], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(ts, qs_norms, label=r'$\|\mathbf{q}(t)\|$')
    axes[0].set_xlabel('$t,s$')
    axes[0].set_ylabel(r'$\|\mathbf{q}\|$')
    axes[0].legend(loc='best')

    axes[1].plot(ts, dqs_norms, label=r'$\|\mathbf{v}(t)\|$')
    axes[1].set_xlabel('$t,s$')
    axes[1].set_ylabel(r'$\|\mathbf{v}\|$')
    axes[1].legend(loc='best')
    fig.savefig('q_and_v_history.png')
    if args.plot:
        plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(ts, diff_poses[:,0], label=r'$x_d(t) - x_s(t)$', color='b')
    axes[0].plot(ts, diff_poses[:,1], label=r'$y_d(t) - y_s(t)$', color='r')
    axes[0].plot(ts, diff_poses[:,2], label=r'$z_d(t) - z_s(t)$', color='g')
    axes[0].plot(ts, np.ones_like(ts) * trend_diff_poses[0], '--', 
                 label=fr'converged $x_d - x_s$ = {trend_diff_poses[0]:.4f}', color='b', alpha=0.5)
    axes[0].plot(ts, np.ones_like(ts) * trend_diff_poses[1], '--', 
                 label=fr'converged $y_d - y_s$ = {trend_diff_poses[1]:.4f}', color='r', alpha=0.5)
    axes[0].plot(ts, np.ones_like(ts) * trend_diff_poses[2], '--', 
                 label=fr'converged $z_d - z_s$ = {trend_diff_poses[2]:.4f}', color='g', alpha=0.5)
    axes[0].set_xlabel('$t,s$')
    axes[0].set_ylabel('meters')
    axes[0].legend(loc='best')

    axes[1].plot(ts, end_poses[:,0], label=r'$x_e(t)$')
    axes[1].plot(ts, end_poses[:,1], label=r'$y_e(t)$')
    axes[1].plot(ts, end_poses[:,2], label=r'$z_e(t)$')
    axes[1].set_xlabel('$t,s$')
    axes[1].set_ylabel('meters')
    axes[1].legend(loc='best')
    fig.savefig('poses_history.png')
    if args.plot:
        plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(ts, diff_lin_vels, 
                 label=r'$\|\boldsymbol{\upsilon}_d(t) - \boldsymbol{\upsilon}_s(t)\|$', 
                 color='b')
    axes[0].plot(ts, diff_ang_vels, 
                 label=r'$\|\boldsymbol{\omega}_d(t) - \boldsymbol{\omega}_s(t)\|$',
                 color='r')
    axes[0].plot(ts, np.ones_like(ts) * trend_diff_lin, '--',
                 label=fr'converged $\|\boldsymbol{{\upsilon}}_d - \boldsymbol{{\upsilon}}_s\|$ = {trend_diff_lin:.4f}',
                 color='b', alpha=0.5)
    axes[0].plot(ts, np.ones_like(ts) * trend_diff_ang, '--',
                 label=fr'converged $\|\boldsymbol{{\omega}}_d - \boldsymbol{{\omega}}_s\|$ = {trend_diff_ang:.4f}',
                 color='r', alpha=0.5)
    axes[0].set_xlabel('$t,s$')
    axes[0].set_ylabel(r'$\frac{m}{s}$ or $\frac{rad}{s}$')
    axes[0].legend(loc='best')

    axes[1].plot(ts, end_lin_vels, label=r'$\|\boldsymbol{\upsilon}_e(t)\|$', color='b')
    axes[1].plot(ts, end_ang_vels, label=r'$\|\boldsymbol{\omega}_e(t)\|$', color='r')
    axes[1].set_xlabel('$t,s$')
    axes[1].set_ylabel(r'$\frac{m}{s}$ or $\frac{rad}{s}$')
    axes[1].legend(loc='best')
    fig.savefig('vels_history.png')
    if args.plot:
        plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(ts, const_errors_norm, label=r'$\|\mathbf{e}_c(t)\|$', color='b')
    axes[0].plot(ts, np.ones_like(ts) * trend_const_e, '--', 
                 label=fr'converged $\|\mathbf{{e}}_c\| = {trend_const_e:.4f}$', 
                 color='b', alpha=0.5)
    axes[0].set_xlabel('$t,s$')
    axes[0].legend(loc='best')

    axes[1].plot(ts, control_errors_norm, label=r'$\|\mathbf{e}_u(t)\|$', color='b')
    axes[1].plot(ts, np.ones_like(ts) * trend_control_e, '--', 
                 label=fr'converged $\|\mathbf{{e}}_u\| = {trend_control_e:.4f}$', 
                 color='b', alpha=0.5)
    axes[1].set_xlabel('$t,s$')
    axes[1].legend(loc='best')
    fig.savefig('errors_history.png')
    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
