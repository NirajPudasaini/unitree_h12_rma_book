
### `sim2mujoco_eval.py`
Evaluation script for Sim2Sim transfer to MuJoCo. Runs trained policies in MuJoCo simulation to verify transfer performance before real hardware deployment. This is a **generic framework** that works with any task by automatically parsing the I/O descriptor YAML file.

The `sim2mujoco` package is borrowed from [WBC Agile](https://github.com/nvidia-isaac/WBC-AGILE) and provides the core functionality for policy transfer and evaluation in MuJoCo.

Example usage:
```bash
python sim2mujoco_eval.py --checkpoint path/to/policy.pt --config path/to/config.yaml --mjcf path/to/robot.xml
```