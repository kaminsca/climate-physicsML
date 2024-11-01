# climate-physicsML

## Configuration

### 0. Download data with rclone

`module load rclone/1.64.0`

`rclone config`

To copy the data into the local_path (Clark's might be different because it might be in the "shared with me" folder):

`rclone copy dropbox:climsim_data local_path --progress`

### 1. Start interactive gpu session
For CSE598_002 account:

`srun --account=cse598s002f24_class --partition=spgpu --gres=gpu:1 --time=2:00:00 --mem-per-cpu=32000m --job-name="CS" --pty bash`

For Mihalcea Lab's account:

`srun --account=mihalcea98 --partition=spgpu --gres=gpu:1 --time=2:00:00 --mem-per-cpu=32000m --job-name="CS" --pty bash`

### 2. Conda configuration

`conda create -n climsim_env python=3.11 -y`

`cd ClimSim`

`pip install .`

### 3. GreatLakes modules

`module load cuda/12.3.0`

`module load cudnn/12.3-v8.9.6`

# Utils

ipython:

`%load_ext autoreload`
`%autoreload 2`
