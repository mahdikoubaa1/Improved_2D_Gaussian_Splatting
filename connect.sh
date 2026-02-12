#!/bin/bash
sshfs -o allow_other,ssh_command='ssh -4' koubaa@ml3d.vc.in.tum.de:/cluster/51/koubaa/mahdi/2DGaussianSplatting .
ssh koubaa@ml3d.vc.in.tum.de  -t 'cd /cluster/51/koubaa/mahdi/2DGaussianSplatting && salloc --gpus=1'vers=4.2