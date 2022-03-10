from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "bayesian_mlp",
    "query": ["random", "entropy", "kcentergreedy", "bald", "batchbald"],
    "data": "toy_moons",
    "active": "toy_two_moons",
}

hparam_dict = {
    # "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 40,
    "trainer.seed": 12345,
    "trainer.vis_callback": True,
    "model.dropout_p": [0, 0.5],
}
naming_conv = (
    "2dtoy/active_basic_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = None

path_to_ex_file = "src/main_toy.py"

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = ExperimentLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    launcher = ExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
