import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

class TeacherStudentVideoMAE(nn.Module):
    def __init__(self, num_classes=33):
        super().__init__()
        config  = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        config.num_frames = 4
        config.num_labels = num_classes
        self.teacher = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-ssv2", # large, frozen
            ignore_mismatched_sizes=True,
            config=config,
            cache_dir="/Data/nadine.hage-chehade"
        )
        
        for p in self.teacher.parameters():
            p.requires_grad = False

        config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
        config.num_frames = 4
        config.num_labels = num_classes
        self.student = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base", # base, trainable
            ignore_mismatched_sizes=True,
            config=config,
            cache_dir="/Data/nadine.hage-chehade"
        )

    def forward(self, x): 
        with torch.no_grad():
            teacher_logits = self.teacher(x).logits       
        return self.student(x).logits, teacher_logits
    
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    """
    alpha  : weight between hard (CE) and soft (KD) loss
    temperature: higher T = softer teacher distribution
                 T=4 works well for SSv2
    """
    # Hard loss: standard cross-entropy with true labels
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=0.1)

    # Soft loss: KL divergence between student and teacher distributions
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits  / temperature, dim=-1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    kd_loss = kd_loss * (temperature ** 2)   # rescale gradient magnitude

    return alpha * ce_loss + (1 - alpha) * kd_loss

def build_optimizer_videomae(model, cfg, backbone_scale=0.1):
    head_params = list(model.student.classifier.parameters())
    backbone_params = [p for n, p in model.student.named_parameters()
                       if 'classifier' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': cfg.training.lr, 'name': 'head'},
        {'params': backbone_params, 'lr': cfg.training.lr * backbone_scale, 'name': 'backbone'},
    ], weight_decay=cfg.training.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs
    )

    return optimizer, scheduler

if __name__ == "__main__":
    vid = torch.randn(2, 4, 3, 224, 224)  # (B, T, C, H, W)
    model = TeacherStudentVideoMAE(num_classes=33)
    student_logits, teacher_logits = model(vid)
    print("Student logits shape:", student_logits.shape)  # (B, num_classes)
    print("Teacher logits shape:", teacher_logits.shape)  # (B, num_classes)