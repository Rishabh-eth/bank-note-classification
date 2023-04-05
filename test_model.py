import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_model(model, test_loader):
    """
    Evaluate the trained model on the test data and calculate the evaluation metrics.

    Args:
        model: Trained PyTorch model.
        test_loader: PyTorch DataLoader object for the test data.

    Returns:
        accuracy: float, calculated accuracy score.
        f1: float, calculated F1 score.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize empty lists for storing true and predicted labels
    true_labels = []
    predicted_labels = []
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for data in test_loader:
            # Get the input images and their corresponding labels
            images, labels = data
            
            # Forward pass through the model
            outputs = model(images)
            
            # Apply sigmoid activation function to the model outputs
            # and set threshold to 0.5 to get predicted labels
            predicted = torch.sigmoid(outputs) > 0.5
            
            # Convert the tensors to lists and extend the true and predicted label lists
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.squeeze(1).tolist())

    # Calculate evaluation metrics using scikit-learn's functions
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return accuracy, f1
