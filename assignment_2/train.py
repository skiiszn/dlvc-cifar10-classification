
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights

from assignment_2_code.models.segment_model import DeepSegmenter
from assignment_2_code.dataset.oxfordpets import  OxfordPetsCustom
from assignment_2_code.metrics import SegMetrics
from assignment_2_code.trainer import ImgSemSegTrainer



def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root="path_to_dataset", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root="path_to_dataset", 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)



    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    num_classes = len(train_data.classes_seg)  # pet, background, border -> 3

    # Pre-trained backbone (Part 4, second run) vs. fully from scratch (first run).
    # We do NOT use `weights=...` because the prediction head has a different
    # number of output classes than the dataset the FCN was pretrained on.
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if args.pretrained else None


    fcn = fcn_resnet50(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        aux_loss=False,
    )

    model = DeepSegmenter(fcn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    #args = argparse.ArgumentParser(description='Training')
    #args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--gpu_id', default='0', type=str,
                        help='index of which GPU to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet-pretrained ResNet50 backbone')
    
    #if not isinstance(args, tuple):
    #    args = args.parse_args()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31


    train(args)