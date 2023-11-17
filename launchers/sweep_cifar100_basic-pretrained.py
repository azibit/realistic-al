from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": [
        "resnet"
    ],  # , "wideresnet-cifar10"], currently there are only pretrained models for resnet18 available!
    "data": "cifar100",
    # "optim": ["sgd"],
    "optim": ["sgd_cosine"],
}

hparam_dict = {
    "trainer.run_test": False,
    "active.num_labelled": [500, 1000, 5000],  # according to FixMatch
    "data.val_size": [2500, None, None],
    "model.dropout_p": [0],
    "model.learning_rate": [0.001, 0.01],  # is more stable than 0.1!
    "model.weight_decay": [5e-3, 5e-4],
    "model.small_head": False,
    "model.use_ema": False,
    "model.finetune": [False],
    "model.freeze_encoder": [False],
    "model.load_pretrained": True,
    "trainer.max_epochs": 80,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugmentMC",
    ],
    "trainer.precision": 16,
    "trainer.batch_size": 1024,
}

joint_iteration = [
    ["model.load_pretrained", "trainer.seed"],
    ["active.num_labelled", "data.val_size"],
]

# naming_conv = "sweep_basic-pretrained_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
naming_conv = "sweep/{data}/basic-pretrained_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}_trafo-{data.transform_train}"

path_to_ex_file = "src/run_training.py"


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    # parser.add_argument("--data", type=str, default=config_dict["data"])
    parser.add_argument("--model", type=str, default=config_dict["model"])
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    # config_dict["data"] = launcher_args.data
    config_dict["model"] = launcher_args.model

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
