
from shutil import copyfile
def refresh():
    copyfile('networks/1chr_digits_model_conv.ckpt.data-00000-of-00001','networks/interf_trained.ckpt.data-00000-of-00001')
    copyfile('networks/1chr_digits_model_conv.ckpt.index','networks/interf_trained.ckpt.index')
    copyfile('networks/1chr_digits_model_conv.ckpt.meta','networks/interf_trained.ckpt.meta')
refresh()