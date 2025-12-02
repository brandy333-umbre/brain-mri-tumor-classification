import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from dataset import BrainTumorDataset
from model import get_model

def evaluate(data_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_ds = BrainTumorDataset(f"{data_dir}/Testing", transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = get_model(4)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            pred_labels = outputs.argmax(1).cpu()
            preds.extend(pred_labels.numpy())
            targets.extend(labels.numpy())

    print(classification_report(targets, preds))
    print(confusion_matrix(targets, preds))

if __name__ == "__main__":
    evaluate("/kaggle/input/mri-brain-tumor-dataset-4-class-7023-images")
