# Trajectory Conditioned guide

## Training：
1. `right_wrist_image`
```
python examples/airbot/convert_airbot_data_to_lerobot.py
python scripts/compute_norm_stats.py --config-name pi0_airbot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_airbot --exp-name=right_wrist_image --overwrite
```




## Deployment
1. always copy the policy and config
