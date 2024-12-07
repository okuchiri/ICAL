from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.utilities.cli import LightningCLI

from ical.datamodule import HMEDatamodule
from ical.lit_ical import LitICAL

def main():
    cli = LightningCLI(
        LitICAL,
        HMEDatamodule,
        save_config_overwrite = True,
        trainer_defaults = {"plugins": SingleDevicePlugin("cuda:0")},
    )


if __name__ == '__main__':
    main()