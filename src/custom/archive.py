def _genDenseModel(model):
  print ("[INFO] Squeezing the sparse model to dense one...")
  conv_layer_keys = ['layer', 'conv', 'weight']
  for name, param in model.named_parameters():
    match = len([x for x in conv_layer_keys if x in name])
    if match == len(conv_layer_keys):
      dims = list(param.shape)

      # Identify zero forced input channels => Remove
      dense_in_ch_idxs = []
      for c in range(dims[1]):
        max_w = torch.max( torch.abs(param[:,c,:,:]) )
        if max_w == 0: 
          dense_in_ch_idxs.append(c)

      # Identify zero forced output channels => Remove
      dense_out_ch_idxs = []
      for c in range(dims[0]):
        max_w = torch.max( torch.abs(param[c,:,:,:]))
        if max_w == 0: 
          dense_out_ch_idxs.append(c)

      num_in_ch = len(dense_in_ch_idxs)
      num_out_ch = len(dense_out_ch_idxs)

      # Generate a new dense tensor and replace
      new_param = torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])
      for in_ch in dense_in_ch_idxs:
        for out_ch in dense_out_ch_idxs:
          new_param.data[in_ch,out_ch,:,:] = param.data[in_ch,out_ch,:,:]

      param.data.copy( new_param.data )

      # Remove the channels in the following BN layer
      layer_name = name.split("conv")[0]
      layer_num = name.split("conv")[1].split(".")[0]

      bn_name_w = layer_name+"bn"+layer_num+".weight"
      bn_name_b = layer_name+"bn"+layer_num+".bias"

      bn_param_w = model.state_dict()[bn_name_w]
      bn_param_b = model.state_dict()[bn_name_b]

      new_bn_w_param = torch.Tensor(num_out_ch)
      new_bn_b_param = torch.Tensor(num_out_ch)
      for out_ch in dense_out_ch_idxs:
        new_bn_w_param[out_ch] = bn_param_w[out_ch]
        new_bn_b_param[out_ch] = bn_param_b[out_ch]

      bn_param_w.data.copy( new_bn_w_param.data )
      bn_param_b.data.copy( new_bn_b_param.data )




def _makeSparse(model, threshold, connection='union'):
  print ("[INFO] Force the sparse filters to zero...")

  dense_chs = {}
  chs_temp = {}
  idx = 0

  for name, param in model.named_parameters():
    dims = list(param.shape)
    if (('conv' in name) or ('fc' in name)) and ('weight' in name):
      dense_in_chs = []
      dense_out_chs = []

      if param.dim() == 4:
        # Forcing input channels to zero
        for c in range(dims[1]):
          if param[:,c,:,:].abs().max() < threshold:
            with torch.no_grad():
              param[:,c,:,:] = 0.
          else:
            dense_in_chs.append(c)
            
        # Forcing output channels to zero
        for c in range(dims[0]):
          if param[c,:,:,:].abs().max() < threshold:
            with torch.no_grad():
              param[c,:,:,:] = 0.
          else:
            dense_out_chs.append(c)

      # Forcing input channels of FC layer to zero
      elif param.dim() == 2:
        for c in range(dims[1]):
          if param[:,c].abs().max() < threshold:
            with torch.no_grad():
              param[:,c] = 0.
          else:
            dense_in_chs.append(c)

        # FC's output channels (class probabilities) are all dense
        dense_out_chs = [c for c in range(dims[0])]

      dense_chs[name] = {'in_chs':dense_in_chs, 'out_chs':dense_out_chs}
      chs_temp[idx]   = {'name':name, 'in_chs':dense_in_chs, 'out_chs':dense_out_chs}
      idx += 1

  # Get union of pre_layer's output channels and cur_layer's input channels
  # Last model reconstruction should be shaped by intersection
  for idx in sorted(chs_temp):
    if idx != 0 and 'weight' in chs_temp[idx]['name']:
      if connection == 'union':
        edge = list(set().union(chs_temp[idx-1]['out_chs'], chs_temp[idx]['in_chs']))

      print ("zeroed [{}] pre_out_chs and [{}] cur_in_chs are maintained".format(
        len(edge) - len(chs_temp[idx-1]['out_chs']), 
        len(edge) - len(chs_temp[idx]['in_chs']))
      )

      dense_chs[ chs_temp[idx-1]['name'] ]['out_chs'] = edge
      dense_chs[ chs_temp[idx]['name'] ]['in_chs'] = edge

  return dense_chs




"""
Return a array of filter sparsity
- Each element in a array represents the MAX(filter matrix data)
"""
def _getConvStructSparsityResNet(model, threshold):
  conv_density = {}
  sparse_map = {}
  conv_layer_keys = ['layer', 'conv', 'weight']
  conv_id = 0

  for name, param in model.named_parameters():
    match = len([x for x in conv_layer_keys if x in name])
    if match == len(conv_layer_keys):
      # Filter sparsity graph: Row(in_chs), Col(out_chs)
      # Tensor dims = [out_chs, in_chs, fil_height, fil_width]
      layer = []
      dims = list(param.shape)
      channel_map = np.zeros([dims[1], dims[0]])

      for in_ch in range(dims[1]):
        fil_row = []
        for out_ch in range(dims[0]):
          fil = param.data.numpy()[out_ch,in_ch,:,:]
          fil_max = np.absolute(fil).max()
          fil_row.append(fil_max)
          if fil_max > threshold:
          #if fil_max > 0.:
            channel_map[in_ch, out_ch] = 1
        layer.append(fil_row)

      sparse_map[conv_id] = np.array(layer)
      #sparse_map[conv_id] = channel_map

      rows = channel_map.max(axis=1) # in_channels
      cols = channel_map.max(axis=0) # out_channels

#      if conv_id != 0:
#        prev_out_ch = sparse_map[conv_id-1].max(axis=0)
#        cur_in_ch = sparse_map[conv_id].max(axis=1)
#        print("==========================")
#        print("out_chs[{}]: {}".format(conv_id, prev_out_ch))
#        print("in_chs[{}]: {}".format(conv_id+1, cur_in_ch))

#      if conv_id != 0:
#        prev_out_ch = sparse_map[conv_id-1].max(axis=0)
#        cur_in_ch = sparse_map[conv_id].max(axis=1)
#        if not np.array_equal(prev_out_ch, cur_in_ch):
#          print("==========================")
#          print("out_chs[{}]: {}".format(conv_id, prev_out_ch))
#          print("in_chs[{}]: {}".format(conv_id+1, cur_in_ch))

      out_density =  float(np.count_nonzero(cols)) / len(cols)
      in_density = float(np.count_nonzero(rows)) / len(rows)
      conv_density[conv_id] = {'in_ch':in_density, 'out_ch':out_density}

      conv_id +=1

  return sparse_map, conv_id, conv_density



"""
Force weights under threshold to zero
- Zero both convolution and normalization layers' parameters to zero
"""
def _makeSparseResNet(model, threshold, norm_zeroing=True, resnet_v2=True):
  print ("[INFO] Force the sparse filters to zero...")

  conv_layer_keys = ['conv', 'weight']

  for name, param in model.named_parameters():
    match = len([x for x in conv_layer_keys if x in name])
    if match == len(conv_layer_keys):
      dims = list(param.shape)

      # Forcing input channels to zero
      for c in range(dims[1]):
        max_w = torch.max( torch.abs(param[:,c,:,:]) )
        if max_w < threshold:
          with torch.no_grad():
            param[:,c,:,:] = 0.

            # Zero next BN layer's channel
            if norm_zeroing and resnet_v2:
              layer_name = name.split("conv")[0]
              layer_num = name.split("conv")[1].split(".")[0]

              bn_name_w = layer_name+"bn"+layer_num+".weight"
              bn_name_b = layer_name+"bn"+layer_num+".bias"

              if bn_name_w in model.state_dict():
                bn_param_w = model.state_dict()[bn_name_w]
                bn_param_b = model.state_dict()[bn_name_b]
                with torch.no_grad():
                  try:
                    bn_param_w[c] = 0.
                    bn_param_b[c] = 0.
                  except:
                    print("======= Error at: {}".format(bn_name_w))

          
      # Forcing output channels to zero
      for c in range(dims[0]):
        max_w = torch.max( torch.abs(param[c,:,:,:]))
        if max_w < threshold:
          with torch.no_grad():
            param[c,:,:,:] = 0.

            # Zero next BN layer's channel
            if norm_zeroing and not resnet_v2:
              layer_name = name.split("conv")[0]
              layer_num = name.split("conv")[1].split(".")[0]

              bn_name_w = layer_name+"bn"+layer_num+".weight"
              bn_name_b = layer_name+"bn"+layer_num+".bias"

              if bn_name_w in model.state_dict():
                bn_param_w = model.state_dict()[bn_name_w]
                bn_param_b = model.state_dict()[bn_name_b]
                with torch.no_grad():
                  bn_param_w[c] = 0.
                  bn_param_b[c] = 0.
