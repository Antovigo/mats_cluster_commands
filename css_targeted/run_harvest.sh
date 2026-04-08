#!/bin/bash

cd ~/SPD/spd_alt

export TMPDIR=/ephemeral/$USER
uv run spd-harvest /mnt/nw/home/a.vigouroux/SPD/batch_commands/css_targeted/harvest_config.yaml
