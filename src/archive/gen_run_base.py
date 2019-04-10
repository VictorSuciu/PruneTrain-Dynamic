#!/work/03883/erhoo/anaconda2/bin/python
import os

model_dir = '/work/03883/erhoo/projects/spar/sparse_train_pytorch/output/cifar10/resnet/resnet50/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

cfg_base = {
    'description':'base',
    'dataset':'cifar100',
    'epochs':240,
    #'epochs':90,
    'start-epoch':0,
    'train_batch':128,
    'test_batch':100,
    'learning-rate':0.1,
    'schedule':'101 151 201',
    'checkpoint':model_dir,
    'resume':False,
    'arch':'resnet50_bt_flat',
    'depth':32,
    'evaluate':False,
    'gpu-id':0,
    'sparse_interval':10,
    'threshold':0.0001,
    'sparse_train':True,
    'norm_zeroing':False,
    'en_auto_lasso_coeff':True,
    'var_auto_lasso_coeff':0.1,
    'reconf_arch':False,
    'arch_out_dir':None,
    'save_checkpoint':10
    }

cfgs = []

cfg = dict(cfg_base)
cfg['description'] = '0.05'
cfg['var_auto_lasso_coeff'] = 0.05
cfgs.append(cfg)


# Generate directories
for cfg in cfgs:
    if not os.path.exists( os.path.join(model_dir, cfg['description']) ):
        os.makedirs( os.path.join(model_dir, cfg['description']) )

out_f = open("./run_files/run_file", 'w')

#======================== Build command line
for cfg in cfgs:
    cmd_line = 'python ../cifar.py '
    cmd_line += ' --dataset '               +cfg['dataset']
    cmd_line += ' --epochs '                +str(cfg['epochs'])
    cmd_line += ' --start-epoch '           +str(cfg['start-epoch'])
    cmd_line += ' --learning-rate '         +str(cfg['learning-rate'])
    cmd_line += ' --schedule '              +cfg['schedule']
    cmd_line += ' --checkpoint '            +os.path.join(cfg['checkpoint'], cfg['description'])
    cmd_line += ' --arch '                  +cfg['arch']
    cmd_line += ' --depth '                 +str(cfg['depth'])
    cmd_line += ' --gpu-id '                +str(cfg['gpu-id'])
    cmd_line += ' --train_batch '           +str(cfg['train_batch'])
    cmd_line += ' --test_batch '            +str(cfg['test_batch'])
    cmd_line += ' --save_checkpoint '       +str(cfg['save_checkpoint'])
    cmd_line += ' --sparse_interval '       +str(cfg['sparse_interval'])
    cmd_line += ' --threshold '             +str(cfg['threshold'])
    cmd_line += ' --var_auto_lasso_coeff '  +str(cfg['var_auto_lasso_coeff'])

    cmd_line += ' --resume '                if cfg['resume'] else ''
    cmd_line += ' --evaluate '              if cfg['evaluate'] else ''
    cmd_line += ' --sparse_train '          if cfg['sparse_train'] else ''
    cmd_line += ' --norm_zeroing '          if cfg['norm_zeroing'] else ''
    cmd_line += ' --en_auto_lasso_coeff '   if cfg['en_auto_lasso_coeff'] else ''
    cmd_line += ' --resnet_v2 '             if cfg['arch'] == 'resnet_v2' else ''
    cmd_line += ' --arch_out_dir '          +cfg['arch_out_dir'] if cfg['reconf_arch'] else ''
    cmd_line += ' >> '                      +os.path.join(cfg['checkpoint'], cfg['description'])+'.log'
    cmd_line += '\n'

    out_f.write(cmd_line)
