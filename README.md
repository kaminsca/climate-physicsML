# climate-physicsML

## Configuration

### 0. Download data with rclone

`module load rclone/1.64.0`

`rclone config`

To copy the data into the local_path (Clark's might be different because it might be in the "shared with me" folder):

`rclone copy dropbox:climsim_data local_path --progress`

### 1. Start interactive gpu session
For CSE598_002 account:

`srun --account=cse598s002f24_class --partition=standard --time=1:00:00 --ntasks=1 --cpus-per-task=16 --mem=32G --job-name="CS" --pty bash`

`srun --account=cse598s002f24_class --partition=spgpu --gres=gpu:1 --time=2:00:00 --mem-per-cpu=32000m --job-name="CS" --pty bash`

For Mihalcea Lab's account with GPU:

`srun --account=mihalcea98 --partition=spgpu --gres=gpu:1 --time=2:00:00 --mem-per-cpu=32000m --job-name="CS" --pty bash`

### 2. Conda configuration

`conda create -n climsim_env python=3.11 -y`

`conda activate climsim_env`

`cd ClimSim`

`pip install .`

`pip install jupyter notebook ipykernel nbformat jupyter_core`

`pip install nvidia-modulus torch` # for ecMLP


### 3. GreatLakes modules

`module load cuda/12.3.0`

`module load cudnn/12.3-v8.9.6`

### 4. Connect to jupyternotebook in greatlakes

Check your node:

`sq`
e.g.: gl1512

`ssh gl1512`

`tmux new -t "cs"`

`conda activate climsim_env`

MYPORT=$(($(($RANDOM % 10000))+49152)); echo $MYPORT
e.g.:
51218

`jupyter notebook --no-browser --port=<MYPORT> --ip=0.0.0.0`
e.g.:
`jupyter notebook --no-browser --port=51218 --ip=0.0.0.0`


`jupyter-notebook --no-browser --port=49941 --ip=0.0.0.0`

# copy link  indicated here into the vscode interactive cell when you hit kernel -> jupyter server
To access the server, open this file in a browser:
        file:///home/alvarovh/.local/share/jupyter/runtime/jpserver-4094100-open.html
    Or copy and paste one of these URLs:
        http://gl1521.arc-ts.umich.edu:49941/tree?token=ef7e9661845603d7a30679abc72c4a25958262bedfc91822  <- THIS ONE
        http://127.0.0.1:49941/tree?token=ef7e9661845603d7a30679abc72c4a25958262bedfc91822

Copy that link into your vscode kernel URL when you open the jupyter notebook

# Utils

ipython:

`%load_ext autoreload`
`%autoreload 2`


