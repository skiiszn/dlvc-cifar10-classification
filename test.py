# Feel free to change the imports according to your implementation and needs
import argparse
import csv
import os
import torch
import torchvision.transforms.v2 as v2
import os
from pathlib import Path

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR



def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use config.py for machine-dependent paths, e.g. DATA_DIR and MODEL_SAVE_DIR.
    test_data = CIFAR10Dataset(DATA_DIR, subset=Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, num_workers=2
)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_data = len(test_data)

    model = DeepClassifier(resnet18(num_classes=10))
    model.load(args.path_to_trained_model)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    # Below implement testing loop and print final loss
    # and metrics to terminal after testing is finished
    # ...
    test_metric.reset()
    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long).to(
                device, non_blocking=True
            )

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            n_batches += 1
            test_metric.update(outputs.detach(), labels)

    test_loss = running_loss / max(n_batches, 1)

    print(f"test loss: {test_loss}")
    print(test_metric)

    # save results
    results_dir = Path("results/test_data")
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.path_to_trained_model).parent.name
    csv_path = results_dir / f"ResNet18_{tag}_test_results.csv"  # change for mutiple testing !!!

    mean_pcacc = test_metric.per_class_accuracy()
    overall_acc = test_metric.accuracy()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_loss", test_loss])
        writer.writerow(["accuracy", overall_acc])
        writer.writerow(["per_class_accuracy", mean_pcacc])
        for classname in test_data.classes:
            writer.writerow(
                [
                    f"accuracy_{classname}",
                    test_metric.per_class_accuracies.get(classname, 0.0),
                ]
            )

    print(f"\nResults saved to {csv_path}")




if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="5", type=str, help="index of which GPU to use"
    )
    args.add_argument("--batch_size", default=128, type=int, help="batch size")
    args.add_argument(
        "--path_to_trained_model",
        type=str,
        default=str(MODEL_SAVE_DIR / "ResNet18_noaug_noreg_adamw_exp" / "best_model_best.pth"),
        help="Path to trained model checkpoint"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    test(args)
