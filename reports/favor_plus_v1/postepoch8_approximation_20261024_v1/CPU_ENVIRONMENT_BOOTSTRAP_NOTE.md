# Isolated CPU diagnostic environment

The system lacks ensurepip; the first venv bootstrap/pip invocation failed.
No system package or shared training environment was modified. The isolated
/tmp environment was completed using the existing pip --python option, with
system-site-packages access for scientific dependencies and a local CPU-only
PyTorch wheel from the official CPU index. This avoids CUDA-linked autograd
engine enumeration during candidate gradient tests. Logs are preserved.
The CPU runtime may differ from the original GPU runtime; frozen-array and
independent-reference comparisons, not historical metric replay, are its purpose.

The first local Torch import also resolved native libraries from the global
CUDA-linked Torch via inherited LD_LIBRARY_PATH, producing a version mismatch.
Only child diagnostic processes now prepend the private CPU wheel torch/lib.
Verified torch.version.cuda=None and no libtorch_cuda in /proc/self/maps; worker
guards retain those checks. No global library or training-environment change.
Recreate with venv --without-pip --system-site-packages and existing pip --python;
retain the documented process-local library path when running diagnostics.
