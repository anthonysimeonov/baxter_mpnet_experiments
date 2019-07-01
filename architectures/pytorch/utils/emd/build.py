import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/my_lib.c']
headers = ['src/my_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/my_lib_cuda.c']
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/emd_cuda.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=["-std=c11"]
)

if __name__ == '__main__':
    ffi.build()
