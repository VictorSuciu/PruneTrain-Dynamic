#!/opt/conda/envs/pytorch-py3.6/bin/python
import os

model_dir = '/work/03883/erhoo/projects/spar/sparse_train_pytorch/output/imagenet'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

dist_args = 'python -m torch.distributed.launch --nproc_per_node=2 '
run_file = 'imagenet_dist.py '

cfg_base = {
    'data':'/work/03883/erhoo/projects/dataset/imagenet-data/raw-data',
    'workers':4,
    'description':'base',
    'dataset':'imagenet',
    'epochs':1,
    'start-epoch':0,
    'train_batch':32,
    'test_batch':32,
    'learning-rate':0.1,
    'schedule':'101 151 201',
    'checkpoint':model_dir,
    'resume':False,
    'arch':'resnet34',
    'depth':50,
    'evaluate':False,
    'gpu-id':'0,1',
    'sparse_interval':10,
    'threshold':0.0001,
    'sparse_train':False,
    'norm_zeroing':False,
    'en_auto_lasso_coeff':False,
    'var_auto_lasso_coeff':0.1,
    'reconf_arch':False,
    'arch_out_dir':None,
    'save_checkpoint':10,
    }

cfgs = []
cfg = dict(cfg_base)
#cfg['arch'] = 'resnet50_flat'
cfg['arch'] = 'resnet50'
cfgs.append(cfg)


#======================== Build command line
def inputArg(flag, arg):
    if isinstance(arg, str):    return ' '+flag+' '+arg
    else:                       return ' '+flag+' '+str(arg)

for cfg in cfgs:
    cmd_line =  dist_args
    cmd_line += run_file
    cmd_line += inputArg('--data',                  cfg['data'])
    cmd_line += inputArg('--workers',               cfg['workers'])
    cmd_line += inputArg('--epochs',                cfg['epochs'])
    cmd_line += inputArg('--start-epoch',           cfg['start-epoch'])
    cmd_line += inputArg('--learning-rate',         cfg['learning-rate'])
    cmd_line += inputArg('--schedule',              cfg['schedule'])
    cmd_line += inputArg('--checkpoint',            os.path.join(cfg['checkpoint'], cfg['description']))
    cmd_line += inputArg('--arch',                  cfg['arch'])
    cmd_line += inputArg('--depth',                 cfg['depth'])
    cmd_line += inputArg('--gpu-id',                cfg['gpu-id'])
    cmd_line += inputArg('--train_batch',           cfg['train_batch'])
    cmd_line += inputArg('--test_batch',            cfg['test_batch'])
    cmd_line += inputArg('--save_checkpoint',       cfg['save_checkpoint'])
    cmd_line += inputArg('--sparse_interval',       cfg['sparse_interval'])
    cmd_line += inputArg('--threshold',             cfg['threshold'])
    cmd_line += inputArg('--var_auto_lasso_coeff',  cfg['var_auto_lasso_coeff'])

    cmd_line += ' --resume '                if cfg['resume'] else ''
    cmd_line += ' --evaluate '              if cfg['evaluate'] else ''
    cmd_line += ' --sparse_train '          if cfg['sparse_train'] else ''
    cmd_line += ' --norm_zeroing '          if cfg['norm_zeroing'] else ''
    cmd_line += ' --en_auto_lasso_coeff '   if cfg['en_auto_lasso_coeff'] else ''
    cmd_line += ' --resnet_v2 '             if cfg['arch'] == 'resnet_v2' else ''
    cmd_line += ' --arch_out_dir '          +cfg['arch_out_dir'] if cfg['reconf_arch'] else ''

    #cmd_line += ' >> '                  +os.path.join(cfg['checkpoint'], cfg['description'])+'.log'

    print (cmd_line)
    os.system(cmd_line)
