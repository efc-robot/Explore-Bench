Quickstart: 
=========================================
.. code-block:: bash
    
    python3.7 -m venv ~/venv/python37_onpolicy
    source ~/venv/python37_onpolicy/bin/activate
    python3 -m pip install --upgrade pip
    pip install setuptools==51.0.0
    pip install wheel
    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    pip install wandb setproctitle absl-py gym matplotlib pandas pygame imageio tensorboardX numba
    # change to the root directory of the onpolicy library
    pip install -e .
 
