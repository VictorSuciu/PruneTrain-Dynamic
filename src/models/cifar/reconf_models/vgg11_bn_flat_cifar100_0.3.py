"""
Flattened VGG11 for CIFAR
"""
                                                                                          
import torch.nn as nn                                                                     
import math                                                                               
                                                                                          
__all__ = ['vgg11_bn_flat']                                                               
                                                                                          
class VGG11(nn.Module):                                                                   
                                                                                          
    # This should be redefined by the channel count                                       
    def __init__(self, num_classes=10):                                                   
        super(VGG11, self).__init__()                                                     
        self.conv1  = nn.Conv2d(3, 25, kernel_size=3, padding=1, bias=False)              
        self.bn1    = nn.BatchNorm2d(25)                                                  
        # MaxPool                                                                         
        self.conv2  = nn.Conv2d(25, 92, kernel_size=3, padding=1, bias=False)             
        self.bn2    = nn.BatchNorm2d(92)                                                  
        # MaxPool                                                                         
        self.conv3  = nn.Conv2d(92, 146, kernel_size=3, padding=1, bias=False)            
        self.bn3    = nn.BatchNorm2d(146)                                                 
        self.conv4  = nn.Conv2d(146, 210, kernel_size=3, padding=1, bias=False)           
        self.bn4    = nn.BatchNorm2d(210)                                                 
        # MaxPool                                                                         
        self.conv5  = nn.Conv2d(210, 224, kernel_size=3, padding=1, bias=False)           
        self.bn5    = nn.BatchNorm2d(224)                                                 
        self.conv6  = nn.Conv2d(224, 173, kernel_size=3, padding=1, bias=False)           
        self.bn6    = nn.BatchNorm2d(173)                                                 
        # MaxPool                                                                         
        self.conv7  = nn.Conv2d(173, 102, kernel_size=3, padding=1, bias=False)           
        self.bn7    = nn.BatchNorm2d(102)                                                 
        self.conv8 = nn.Conv2d(102, 125, kernel_size=3, padding=1, bias=False)            
        self.bn8   = nn.BatchNorm2d(125)                                                  
        self.fc     = nn.Linear(125, num_classes)                                         
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)                               
        self.relu   = nn.ReLU(inplace=True)                                               
                                                                                          
        # parameter initialization                                                        
        for m in self.modules():                                                          
            if isinstance(m, nn.Conv2d):                                                  
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels                  
                m.weight.data.normal_(0, math.sqrt(2. / n))                               
            elif isinstance(m, nn.BatchNorm2d):                                           
                m.weight.data.fill_(1)                                                    
                m.bias.data.zero_()                                                       
                                                                                          
    # This part of architecture remains the same                                          
    def forward(self, x):                                                                 
        x = self.conv1(x)                                                                 
        x = self.bn1(x)                                                                   
        x = self.relu(x)                                                                  
                                                                                          
        x = self.pool(x)                                                                  
                                                                                          
        x = self.conv2(x)                                                                 
        x = self.bn2(x)                                                                   
        x = self.relu(x)                                                                  
        x = self.pool(x)                                                                  
                                                                                          
        x = self.conv3(x)                                                                 
        x = self.bn3(x)                                                                   
        x = self.relu(x)                                                                  
        x = self.conv4(x)                                                                 
        x = self.bn4(x)                                                                   
        x = self.relu(x)                                                                  
        x = self.pool(x)                                                                  
                                                                                          
        x = self.conv5(x)                                                                 
        x = self.bn5(x)                                                                   
        x = self.relu(x)                                                               
        x = self.conv6(x)                                                              
        x = self.bn6(x)                                                                
        x = self.relu(x)                                                               
        x = self.pool(x)                                                               
                                                                                       
        x = self.conv7(x)                                                              
        x = self.bn7(x)                                                                
        x = self.relu(x)                                                               
        x = self.conv8(x)                                                              
        x = self.bn8(x)                                                                
        x = self.relu(x)                                                               
        x = self.pool(x)                                                               
                                                                                       
        x = x.view(x.size(0), -1)                                                      
        x = self.fc(x)                                                                 
        return x                                                                       
                                                                                       
def vgg11_bn_flat(**kwargs):                                                           
    model = VGG11(**kwargs)                                                            
    return model 