from typing import Optional, Any  
import torch  
import torch.nn as nn  
from torchvision.models import resnet50  
from torchvision.transforms import Compose, Resize, ToTensor, Normalize  
import pandas as pd  
from PIL import Image  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import f1_score, accuracy_score, recall_score  
from tqdm import tqdm  
from torch.utils.data import Dataset, DataLoader  

# Define custom types  
ImagePath = str  
Label = int  
BatchData = tuple[torch.Tensor, torch.Tensor]  

# Read CSV data containing image paths and labels  
# Negative samples (label 0) and positive samples (label 1)  
df_0 = pd.read_csv('/home/zhl/Project/Disease/IBD/Data/filtered_no_medicine_china_gut_human_IBD_negative_label_img_path.csv')  
df_1 = pd.read_csv('/home/zhl/Project/Disease/IBD/Data/filtered_no_medicine_china_gut_human_IBD_positive_label_img_path.csv')  

# Concatenate both DataFrames and shuffle the data  
df = pd.concat([df_0, df_1], ignore_index=True)  
df = df[df['image_path'].notnull()]  # Remove rows with null image paths  
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data  

# Split data into training and validation sets while maintaining class distribution  
X: list[ImagePath] = df['image_path'].tolist()  
y: list[Label] = df['label'].tolist()  
X_train, X_val, y_train, y_val = train_test_split(  
    X, y,   
    test_size=0.2,   
    random_state=42,   
    stratify=y  # Ensure same class distribution in train and val sets  
)  

# Define image preprocessing pipeline  
# The normalization values are calculated from the dataset statistics  
transform = Compose([  
    Resize((224, 224)),  # Resize images to standard input size  
    ToTensor(),          # Convert PIL Image to tensor and scale to [0,1]  
    Normalize(mean=[0.0178, 0.0178, 0.0178], std=[0.0873, 0.0873, 0.0873])  # Normalize using dataset statistics  
])  

class CustomDataset(Dataset):  
    """Custom dataset class for loading IBD images and labels."""  
    
    def __init__(self,   
                 image_paths: list[ImagePath],   
                 labels: list[Label],   
                 transform: Optional[Any] = None) -> None:  
        """  
        Initialize the dataset.  
        
        Args:  
            image_paths: List of paths to image files  
            labels: List of corresponding labels  
            transform: Optional transform to be applied to images  
        """  
        self.image_paths = image_paths  
        self.labels = labels  
        self.transform = transform  

    def __len__(self) -> int:  
        """Return the total number of samples in the dataset."""  
        return len(self.image_paths)  

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Label]:  
        """  
        Get a single sample from the dataset.  
        
        Args:  
            index: Index of the sample to fetch  
            
        Returns:  
            Tuple of (transformed_image, label)  
        """  
        image_path = self.image_paths[index]  
        image = Image.open(image_path).convert('RGB')  
        if self.transform:  
            image = self.transform(image)  
        return image, self.labels[index]  

# Create data loaders for training and validation  
train_dataset = CustomDataset(X_train, y_train, transform=transform)  
val_dataset = CustomDataset(X_val, y_val, transform=transform)  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

# Initialize ResNet50 model  
model = resnet50(weights=None)  
# Load pre-trained weights  
model.load_state_dict(torch.load('/home/zhl/regression_and_prediction/resnet50-0676ba61.pth'))  

# Modify the final fully connected layer for binary classification  
num_features = model.fc.in_features  
model.fc = nn.Linear(num_features, 2)  

# Set up training parameters  
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")  
model.to(device)  
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  

def train_epoch(model: nn.Module,   
                loader: DataLoader,   
                criterion: nn.Module,   
                optimizer: torch.optim.Optimizer,   
                device: torch.device) -> tuple[float, float, float, float]:  
    """  
    Train the model for one epoch.  
    
    Args:  
        model: The neural network model  
        loader: DataLoader for training data  
        criterion: Loss function  
        optimizer: Optimization algorithm  
        device: Device to run the training on  
        
    Returns:  
        Tuple of (average_loss, f1_score, accuracy, recall)  
    """  
    model.train()  
    total_loss = 0.0  
    predictions: list[int] = []  
    labels: list[int] = []  
    
    with tqdm(loader, desc="Training", unit="batch") as pbar:  
        for inputs, batch_labels in pbar:  
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)  
            
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, batch_labels)  
            loss.backward()  
            optimizer.step()  
            
            total_loss += loss.item()  
            predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())  
            labels.extend(batch_labels.cpu().tolist())  
            
            pbar.set_postfix(loss=total_loss / len(pbar))  
            
    avg_loss = total_loss / len(loader)  
    return (  
        avg_loss,  
        f1_score(labels, predictions),  
        accuracy_score(labels, predictions),  
        recall_score(labels, predictions)  
    )  

def validate(model: nn.Module,   
            loader: DataLoader,   
            criterion: nn.Module,   
            device: torch.device) -> tuple[float, float, float, float]:  
    """  
    Validate the model.  
    
    Args:  
        model: The neural network model  
        loader: DataLoader for validation data  
        criterion: Loss function  
        device: Device to run the validation on  
        
    Returns:  
        Tuple of (average_loss, f1_score, accuracy, recall)  
    """  
    model.eval()  
    total_loss = 0.0  
    predictions: list[int] = []  
    labels: list[int] = []  
    
    with torch.no_grad():  
        with tqdm(loader, desc="Validating", unit="batch") as pbar:  
            for inputs, batch_labels in pbar:  
                inputs, batch_labels = inputs.to(device), batch_labels.to(device)  
                outputs = model(inputs)  
                loss = criterion(outputs, batch_labels)  
                
                total_loss += loss.item()  
                predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())  
                labels.extend(batch_labels.cpu().tolist())  
                
                pbar.set_postfix(loss=total_loss / len(pbar))  
                
    avg_loss = total_loss / len(loader)  
    return (  
        avg_loss,  
        f1_score(labels, predictions),  
        accuracy_score(labels, predictions),  
        recall_score(labels, predictions)  
    )  

# Training loop  
num_epochs = 10  
best_val_accuracy = 0.0  
best_model_state: dict[str, torch.Tensor] | None = None  

for epoch in range(num_epochs):  
    # Train and validate for one epoch  
    train_loss, train_f1, train_acc, train_recall = train_epoch(  
        model, train_loader, criterion, optimizer, device  
    )  
    val_loss, val_f1, val_acc, val_recall = validate(  
        model, val_loader, criterion, device  
    )  
    
    # Print metrics  
    print(f"Epoch [{epoch+1}/{num_epochs}]")  
    print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, "  
          f"Acc: {train_acc:.4f}, Recall: {train_recall:.4f}")  
    print(f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, "  
          f"Acc: {val_acc:.4f}, Recall: {val_recall:.4f}")  
    
    # Save best model  
    if val_acc > best_val_accuracy:  
        best_val_accuracy = val_acc  
        best_model_state = model.state_dict()  

def save_misclassified_examples(model: nn.Module,   
                              loader: DataLoader,   
                              image_paths: list[str],   
                              device: torch.device) -> None:  
    """  
    Identify and save misclassified examples.  
    
    Args:  
        model: The trained model  
        loader: DataLoader for validation data  
        image_paths: List of image paths  
        device: Device to run the inference on  
    """  
    model.eval()  
    misclassified: dict[str, list] = {  
        'image_path': [],  
        'true_label': [],  
        'predicted_label': []  
    }  
    
    with torch.no_grad():  
        for i, (inputs, labels) in enumerate(loader):  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            predictions = torch.argmax(outputs, dim=1)  
            
            # Find misclassified examples  
            mask = predictions != labels  
            if mask.any():  
                batch_start = i * loader.batch_size  
                for j, is_wrong in enumerate(mask):  
                    if is_wrong:  
                        idx = batch_start + j  
                        misclassified['image_path'].append(image_paths[idx])  
                        misclassified['true_label'].append(labels[j].item())  
                        misclassified['predicted_label'].append(predictions[j].item())  
    
    # Save results  
    pd.DataFrame(misclassified).to_csv(  
        '/home/zhl/Project/Disease/IBD/CNN/ResNet/ResNet50/2024-11-12/misclassified_images.csv',   
        index=False  
    )  

# Load best model and analyze misclassifications  
if best_model_state is not None:  
    model.load_state_dict(best_model_state)  
    save_misclassified_examples(model, val_loader, X_val, device)  
    
    # Save the best model  
    torch.save(  
        best_model_state,  
        '/home/zhl/Project/Disease/IBD/CNN/ResNet/ResNet50/2024-11-12/resnet50_on_filtered_IBD_best.pth'  
    )
