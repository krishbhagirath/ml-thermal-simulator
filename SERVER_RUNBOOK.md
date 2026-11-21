# DGX server runbook

Commands are single-line and use absolute paths. Long jobs run in tmux. Use one GPU and check that it's free first. Never upgrade torch.

## 1. Environment smoke test (not yet run; needed before the first GPU job)
Run each line separately and send back the full output.
1. `conda env list && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv && ldd --version | head -1 && df -h ~ && nproc && free -g`
2. `conda activate <ENV_WITH_PY312_TORCH210>`
3. `CUDA_VISIBLE_DEVICES=<FREE_GPU_INDEX> python -c "import sys,platform,torch;print(sys.version.split()[0],platform.libc_ver());print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0));x=torch.randn(4,64,192,160,device='cuda');y=torch.fft.irfft2(torch.fft.rfft2(x),s=(192,160));print('fft err',(x-y).abs().max().item())"`
4. `python -c "import numpy,scipy,matplotlib;print(numpy.__version__,scipy.__version__,matplotlib.__version__)" ; python -c "import h5py;print('h5py',h5py.__version__)" || echo "h5py missing"`
5. `(mkdir -p /raid/$USER/thermal_write_test && touch /raid/$USER/thermal_write_test/probe && echo RAID_WRITABLE && df -h /raid && rm /raid/$USER/thermal_write_test/probe && rmdir /raid/$USER/thermal_write_test) || echo RAID_NOT_WRITABLE`
6. `pip list 2>/dev/null | grep -i -E "^(torch|triton|numpy|scipy|h5py) "`

## 2. Project setup on the server (after the smoke test)
To be filled in once the smoke-test results are known: the git clone of the public repo on branch `dev`, the venv, the constraints file, and the install.

## 3. Jobs
None yet. The H-net training job (M4) goes here if local training is too slow.
