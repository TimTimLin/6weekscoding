def train_one_epoch(model, dataloader, loss_fn, optimizer)->float:
    model.train()  # Set the model to training mode
    total_loss = 0.0
    for batch_images, batch_labels in dataloader:
        optimizer.zero_grad()  # Clear gradients
        logits = model(batch_images)  # Forward pass
        loss = loss_fn(logits, batch_labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters   
        total_loss += loss.item()*batch_images.size(0)  # Accumulate loss
    epoch_loss = total_loss / len(dataloader.dataset)  # Compute average loss for the epoch
    return epoch_loss