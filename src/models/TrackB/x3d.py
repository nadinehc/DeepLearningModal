import torch
import torch.nn as nn

import torch.nn.functional as F

class X3D(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.model_name = 'x3d_xs'
        self.model = torch.hub.load('facebookresearch/pytorchvideo', self.model_name, pretrained=pretrained)
        
        in_features = self.model.blocks[5].proj.in_features # type: ignore
        self.model.blocks[5].proj = nn.Linear(in_features, 33)  # type: ignore

        nn.init.trunc_normal_(self.model.blocks[5].proj.weight, std=0.02) # type: ignore
        nn.init.constant_(self.model.blocks[5].proj.bias, 0) # type: ignore

        # Freeze blocks 0 through 4 (the entire backbone)
        for i in range(5): 
            for param in self.model.blocks[i].parameters(): # type: ignore
                param.requires_grad = False

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=(x.shape[2], 160, 160), mode='area')
        return self.model(x) # type: ignore
    
def build_x3d_optimizer(model, lr, epochs):
    # Only pass parameters where requires_grad is True
    optimizer = optimizer = torch.optim.AdamW([
        {'params': model.model.blocks[0:5].parameters(), 'lr': lr/100}, # Backbone
        {'params': model.model.blocks[5].parameters(), 'lr': lr}    # New Head
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler

if __name__ == "__main__":
    vid = torch.rand(1, 4, 3, 224, 224)

    model = X3D(True)
    preds = model(vid)

    print(preds.shape)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    top_prob, top_idx = preds.topk(k=1)
    print(f"Predicted class ID: {top_idx.item()}")