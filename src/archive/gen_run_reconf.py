#!/work/03883/erhoo/anaconda2/bin/python
import os

model_dir = '/work/03883/erhoo/projects/spar/sparse_train_pytorch/output/imagenet/resnet50/'
arch_dir = '/work/03883/erhoo/projects/spar/sparse_train_pytorch/models/imagenet'

cfg_base = {
    'description':'base',
    'data':'/work/03883/erhoo/projects/dataset/imagenet-data/raw-data',
    'epochs':90,
    'start-epoch':0,
    'train_batch':256,
    'test_batch':100,
    'learning-rate':0.1,
    'schedule':'41 61 81',
    'checkpoint':model_dir,
    'resume':'',
    'arch':'',
    'depth':32,
    'evaluate':False,
    'gpu-id':'0,1,2,3',
    'sparse_interval':10,
    'threshold':0.0001,
    'en_group_lasso':True,
    'var_group_lasso_coeff':0.1,
    'reconf_arch':True,
    'arch_out_dir':None,
    'save_checkpoint':10
    }

cfgs = []

cfg = dict(cfg_base)
cfg['description'] = '0.05_gtx'
cfg['var_group_lasso_coeff'] = 0.05
cfg['arch'] = 'resnet50_flat_005'
cfg['arch_name'] = cfg['arch']+'_'+cfg['description']
cfg['arch_out_dir'] = os.path.join(model_dir, 'arch', cfg['description'])
cfgs.append(cfg)


# Generate directories
for cfg in cfgs:
    if not os.path.exists( os.path.join(model_dir, cfg['description']) ):
        os.makedirs( os.path.join(model_dir, cfg['description']) )

out_f = open("./run_files/run_reconf_005", 'w')

#======================== Build command line
for cfg in cfgs:
    # Reset architecture
    ori_net = os.path.join(arch_dir, 'resnet50_flat_ori.py')
    target_net = os.path.join(arch_dir, cfg['arch']+'.py')
    cp_net = 'cp {} {}\n'.format(ori_net, target_net)
    print (cp_net)
    #os.system(cp_net)
    out_f.write(cp_net)

    for cur_epoch in range(0, cfg['epochs'], cfg['sparse_interval']):
        cmd_line = 'python ../imagenet_reconf_fp32.py '
        cmd_line += ' --data '                  +cfg['data']
        cmd_line += ' --learning-rate '         +str(cfg['learning-rate'])
        cmd_line += ' --schedule '              +cfg['schedule']
        cmd_line += ' --checkpoint '            +os.path.join(cfg['checkpoint'], cfg['description'])
        cmd_line += ' --arch '                  +cfg['arch']
        cmd_line += ' --gpu-id '                +str(cfg['gpu-id'])
        cmd_line += ' --train_batch '           +str(cfg['train_batch'])
        cmd_line += ' --test_batch '            +str(cfg['test_batch'])
        cmd_line += ' --save_checkpoint '       +str(cfg['save_checkpoint'])
        cmd_line += ' --threshold '             +str(cfg['threshold'])
        cmd_line += ' --evaluate '              if cfg['evaluate'] else ''
        cmd_line += ' --start-epoch '           +str(cfg['start-epoch'])
        cmd_line += ' --epochs '                +str(cur_epoch + cfg['sparse_interval'])

        # Group lasso config
        cmd_line += ' --sparse_interval '       +str(cfg['sparse_interval'])
        cmd_line += ' --var_group_lasso_coeff ' +str(cfg['var_group_lasso_coeff'])
        cmd_line += ' --en_group_lasso '        if cfg['en_group_lasso'] else ''
    
        # Reconfiguration 
        cmd_line += ' --resume '                +cfg['resume'] if cfg['resume'] != '' else ''
        cmd_line += ' --arch_name '             +cfg['arch_name']+'_'+str(cur_epoch)+'.py'
        cmd_line += ' --arch_out_dir '          +cfg['arch_out_dir'] if cfg['reconf_arch'] else ''
        cmd_line += ' >> '                      +os.path.join(cfg['checkpoint'], cfg['description'])+'.log'
        cmd_line += '\n'

        out_f.write(cmd_line)

        checkpoint = 'checkpoint.'+str(cur_epoch+cfg['sparse_interval'])
        cfg['resume'] = os.path.join(model_dir, cfg['description'], checkpoint+'.tar')
