from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time


# class LabelSmoothing(nn.Module):
#     "Implement label smoothing."
    
#     def __init__(self, size, padding_idx=0, smoothing=0.0):
#         super(LabelSmoothing, self).__init__()
#         self.criterion = nn.KLDivLoss(size_average=False)
#         self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.size = size
#         self.true_dist = None
        
#     def forward(self, x, target):
#         assert x.size(1) == self.size
#         true_dist = x.data.clone()
#         true_dist.fill_(self.smoothing / (self.size - 2))
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         true_dist[:, self.padding_idx] = 0
#         mask = torch.nonzero(target.data == self.padding_idx)
#         if mask.dim() > 0:
#             true_dist.index_fill_(0, mask.squeeze(), 0.0)
#         self.true_dist = true_dist
#         return self.criterion(x, Variable(true_dist, requires_grad=False))

# def test(model, tokenizer, dataloader, device, norm_accentuation=False, norm_punctuation=False):
#     """
#     Evaluate and predict model with the test dataloader.
    
#     Args:
#         model: The OCR model to evaluate.
#         tokenizer: Tokenizer for decoding output tokens.
#         dataloader: DataLoader for test dataset.
#         device: Device to run the model on.
#         norm_accentuation: Whether to discard accentuation marks in evaluation.
#         norm_punctuation: Whether to discard punctuation marks in evaluation.
    
#     Returns:
#         None. Prints Character Error Rate, Word Error Rate, and Sequence Error Rate.
#     """
#     model.eval()
#     predicts = []
#     gt = []
#     with torch.no_grad():
#         for batch in dataloader:
#             src, trg = batch
#             src, trg = src.to(device), trg.to(device)
            
#             # Forward pass through model
#             output = model(src.float(), trg.long()[:, :-1])
            
#             out_indexes = []
#             for i in range(output.size(0)):
#                 out_token = output[i].argmax().item()
#                 if out_token == tokenizer.chars.index('EOS'):
#                     break
#                 out_indexes.append(out_token)
            
#             predicts.append(tokenizer.decode(out_indexes))
#             gt.append(tokenizer.decode(trg.flatten(0,1)))
    
#     predicts = list(map(lambda x : x.replace('SOS','').replace('EOS',''), predicts))
#     gt = list(map(lambda x : x.replace('SOS','').replace('EOS',''), gt))
    
#     evaluate = __import__('data.evaluation', fromlist=['ocr_metrics']).ocr_metrics(
#         predicts=predicts,
#         ground_truth=gt,
#         norm_accentuation=norm_accentuation,
#         norm_punctuation=norm_punctuation
#     )
#     print("Calculate Character Error Rate {}, Word Error Rate {} and Sequence Error Rate {}".format(evaluate[0], evaluate[1], evaluate[2]))




def train(model, criterion, optimizer, scheduler, dataloader, vocab_length, device):
    """
    Train the model using the provided dataloader.
    
    Args:
        model: The OCR model
        criterion: Loss function (CTC)
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        dataloader: Training data loader
        vocab_length: Size of vocabulary
        device: Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_items = 0
    
    for batch, (imgs, labels_y,) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels_y = labels_y.to(device)
        batch_size = imgs.size(0)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(imgs.float())
        
        # Ensure output is in (batch, seq_len, vocab_size) format
        if output.dim() != 3:
            raise ValueError(f"Expected 3D output tensor, got shape: {output.shape}")
        
        # Permute to (seq_len, batch, vocab_size) for CTC loss
        log_probs = F.log_softmax(output, dim=2).permute(1, 0, 2)
        
        # Calculate input sequence lengths (all are same length after CNN processing)
        input_lengths = torch.full((batch_size,), 
                                 log_probs.size(0), 
                                 dtype=torch.long,
                                 device=device)
        
        # Calculate target lengths (excluding padding)
        target_lengths = []
        labels_list = []
        valid_samples = []
        
        # Process each sequence in the batch
        for i in range(batch_size):
            # Find non-zero elements (non-padding)
            non_zero = labels_y[i].nonzero().squeeze()
            if non_zero.dim() == 0:  # Handle case of empty sequence
                continue  # Skip this sample
            
            length = non_zero.shape[0]
            if length > log_probs.size(0):  # Skip if target is longer than output
                continue
                
            sequence = labels_y[i, :length]
            target_lengths.append(length)
            labels_list.append(sequence)
            valid_samples.append(i)
        
        # Skip batch if no valid samples
        if not valid_samples:
            continue
            
        # Keep only valid samples
        valid_samples = torch.tensor(valid_samples, device=device)
        log_probs = log_probs[:, valid_samples, :]
        input_lengths = input_lengths[valid_samples]
        
        # Convert target lengths to tensor
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
        
        # Concatenate all label sequences
        labels_packed = torch.cat(labels_list)
        
        # CTC loss calculation
        try:
            loss = criterion(log_probs,  # (T, N, C)
                           labels_packed,  # Flattened target sequences
                           input_lengths,  # Length of each input sequence (N,)
                           target_lengths)  # Length of each target sequence (N,)
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * len(valid_samples)
            total_items += len(valid_samples)
            
        except RuntimeError as e:
            print(f"Error in batch {batch}:")
            print(f"log_probs shape: {log_probs.shape}")
            print(f"labels_packed shape: {labels_packed.shape}")
            print(f"input_lengths shape: {input_lengths.shape}")
            print(f"target_lengths shape: {target_lengths.shape}")
            raise e
    
    return total_loss / total_items if total_items > 0 else float('inf')

def evaluate(model, criterion, dataloader, vocab_length, device):
    """
    Evaluate the model using the provided dataloader.
    
    Args:
        model: The OCR model
        criterion: Loss function (CTC)
        dataloader: Validation data loader
        vocab_length: Size of vocabulary
        device: Device to evaluate on
        
    Returns:
        float: Average loss for the epoch
    """
    model.eval()
    total_loss = 0
    total_items = 0

    with torch.no_grad():
        for batch, (imgs, labels_y) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)
            batch_size = imgs.size(0)

            # Forward pass
            output = model(imgs.float())
            
            # Ensure output is in (batch, seq_len, vocab_size) format
            if output.dim() != 3:
                raise ValueError(f"Expected 3D output tensor, got shape: {output.shape}")
            
            # Permute to (seq_len, batch, vocab_size) for CTC loss
            log_probs = F.log_softmax(output, dim=2).permute(1, 0, 2)
            
            # Calculate input sequence lengths (all are same length after CNN processing)
            input_lengths = torch.full((batch_size,), 
                                     log_probs.size(0), 
                                     dtype=torch.long,
                                     device=device)
            
            # Calculate target lengths (excluding padding)
            target_lengths = []
            labels_list = []
            valid_samples = []
            
            # Process each sequence in the batch
            for i in range(batch_size):
                # Find non-zero elements (non-padding)
                non_zero = labels_y[i].nonzero().squeeze()
                if non_zero.dim() == 0:  # Handle case of empty sequence
                    continue  # Skip this sample
                
                length = non_zero.shape[0]
                if length > log_probs.size(0):  # Skip if target is longer than output
                    continue
                    
                sequence = labels_y[i, :length]
                target_lengths.append(length)
                labels_list.append(sequence)
                valid_samples.append(i)
            
            # Skip batch if no valid samples
            if not valid_samples:
                continue
                
            # Keep only valid samples
            valid_samples = torch.tensor(valid_samples, device=device)
            log_probs = log_probs[:, valid_samples, :]
            input_lengths = input_lengths[valid_samples]
            
            # Convert target lengths to tensor
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            
            # Concatenate all label sequences
            labels_packed = torch.cat(labels_list)
            
            try:
                # CTC loss calculation
                loss = criterion(log_probs,
                               labels_packed,
                               input_lengths,
                               target_lengths)
                
                total_loss += loss.item() * len(valid_samples)
                total_items += len(valid_samples)
                
            except RuntimeError as e:
                print(f"Error in batch {batch}:")
                print(f"log_probs shape: {log_probs.shape}")
                print(f"labels_packed shape: {labels_packed.shape}")
                print(f"input_lengths shape: {input_lengths.shape}")
                print(f"target_lengths shape: {target_lengths.shape}")
                raise e

    return total_loss / total_items if total_items > 0 else float('inf')

def get_memory(model, imgs):
    """
    Extract features and apply positional encoding for the BiLSTM model.
    
    Args:
        model: The OCR model
        imgs: Input images tensor
        
    Returns:
        Memory tensor with shape (seq_len, batch, hidden_dim*2)
    """
    with torch.no_grad():
        # Extract CNN features
        features = model.get_feature(imgs)
        
        # Apply conv layer
        conv_out = model.conv(features)
        
        # Get spatial dimensions
        bs, c, h, w = conv_out.size()
        
        # Add positional encodings
        row_emb = model.row_embed[:h].unsqueeze(1).repeat(1, w, 1)  # (H, W, hidden_dim//2)
        col_emb = model.col_embed[:w].unsqueeze(0).repeat(h, 1, 1)  # (H, W, hidden_dim//2)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1, hidden_dim, H, W)
        pos_emb = pos_emb.to(conv_out.device)
        conv_out = conv_out + pos_emb
        
        # Flatten spatial dimensions and permute for LSTM
        lstm_input = conv_out.flatten(2).permute(0, 2, 1)  # (batch, seq_len, feature)
        
        # Add positional encoding to LSTM input
        lstm_input = lstm_input.permute(1, 0, 2)  # (seq_len, batch, feature)
        lstm_input = model.query_pos(lstm_input)
        lstm_input = lstm_input.permute(1, 0, 2)  # (batch, seq_len, feature)
        
        # Apply BiLSTM
        lstm_out, _ = model.lstm(lstm_input)
        
        # Return in shape (seq_len, batch, hidden_dim*2)
        return lstm_out.permute(1, 0, 2)

def single_image_inference(model, img, tokenizer, transform, device):
    """
    Run inference on single image using greedy decoding.
    
    Args:
        model: The OCR model
        img: Input image
        tokenizer: Tokenizer for encoding/decoding text
        transform: Image transform pipeline
        device: Device to run inference on
        
    Returns:
        pred_text: Predicted text string
    """
    model.eval()
    
    # Preprocess image
    img = transform(img)
    imgs = img.unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        # Forward pass
        output = model(imgs)
        
        # Ensure output is in (seq_len, batch, vocab_size) format
        if output.dim() == 3:
            output = output.permute(1, 0, 2)
        
        # Apply log softmax and get predictions
        output = F.log_softmax(output, dim=2)
        output = output.argmax(dim=2)
        output = output.squeeze(1)  # Remove batch dimension
        
        # Convert prediction to text (handle special tokens)
        out_indices = []
        for idx in output:
            token = idx.item()
            if token == tokenizer.chars.index('EOS'):
                break
            if token > tokenizer.chars.index('EOS'):  # Skip special tokens
                out_indices.append(token)
        
        # Decode the prediction
        pred_text = tokenizer.decode(out_indices)
    
    return pred_text

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, tokenizer, target_path, device):
    """
    Run training for specified number of epochs.
    
    Args:
        model: The OCR model
        criterion: Loss function (CTC)
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        tokenizer: Tokenizer for encoding/decoding text
        target_path: Path to save model checkpoints
        device: Device to train on
    """
    best_valid_loss = float('inf')
    patience = 0
    max_patience = 4  # Number of epochs to wait before reducing learning rate
    
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1:02} | Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        start_time = time.time()
        
        # Training phase
        train_loss = train(model, criterion, optimizer, scheduler, 
                          train_loader, tokenizer.vocab_size, device)
        
        # Validation phase
        valid_loss = evaluate(model, criterion, val_loader, tokenizer.vocab_size, device)
        
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        
        # Save best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss,
            }, target_path)
            patience = 0
        else:
            patience += 1
        
        # Reduce learning rate if validation loss hasn't improved
        if patience >= max_patience:
            scheduler.step()
            patience = 0
        
        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')
        print(f'Best Val Loss: {best_valid_loss:.3f}')
    
    print(f'Training completed. Best validation loss: {best_valid_loss:.3f}')



def test(model, test_loader, max_text_length, tokenizer):
    """
    Evaluate and predict model with the test dataloader.
    Memory-efficient version that processes data in chunks.
    
    Args:
        model: The OCR model to evaluate.
        test_loader: DataLoader for test dataset.
        max_text_length: Maximum length of output sequence.
        tokenizer: Tokenizer for decoding output tokens.
    
    Returns:
        predicts: List of predicted text sequences.
        gt: List of ground truth text sequences.
        imgs: List of input images.
    """
    model.eval()
    predicts = []
    gt = []
    imgs = []
    device = next(model.parameters()).device
    
    # Clear memory before starting
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    chunk_size = 10  # Process 10 samples at a time
    current_chunk = {'imgs': [], 'preds': [], 'gts': []}
    
    with torch.no_grad():
        try:
            for batch_idx, batch in enumerate(test_loader):
                src, trg = batch
                
                # Store CPU version of image
                current_chunk['imgs'].append(src.flatten(0,1).cpu())
                
                # Move tensors to device
                src = src.to(device)
                trg = trg.to(device)
                
                try:
                    # Forward pass without teacher forcing
                    output = model(src.float())
                    
                    # Free memory
                    del src
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Ensure output is in (seq_len, batch, vocab_size) format
                    if output.dim() == 3:
                        output = output.permute(1, 0, 2)
                    
                    # Apply log softmax and get predictions
                    output = F.log_softmax(output, dim=2)
                    predictions = output.argmax(dim=2)
                    
                    # Free memory
                    del output
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Process each sequence in the batch
                    for pred, target in zip(predictions.transpose(0,1), trg):
                        # Convert prediction to text (handle special tokens)
                        pred_indices = []
                        for idx in pred:
                            token = idx.item()
                            if token == tokenizer.chars.index('EOS'):
                                break
                            if token > tokenizer.chars.index('EOS'):  # Skip special tokens
                                pred_indices.append(token)
                        
                        # Decode prediction
                        pred_text = tokenizer.decode(pred_indices)
                        pred_text = pred_text.replace('SOS', '').replace('EOS', '')
                        current_chunk['preds'].append(pred_text)
                        
                        # Convert target to text (handle special tokens)
                        target_indices = []
                        for idx in target:
                            token = idx.item()
                            if token == tokenizer.chars.index('EOS'):
                                break
                            if token > tokenizer.chars.index('EOS'):  # Skip special tokens
                                target_indices.append(token)
                        
                        # Decode target
                        target_text = tokenizer.decode(target_indices)
                        target_text = target_text.replace('SOS', '').replace('EOS', '')
                        current_chunk['gts'].append(target_text)
                    
                    # Free memory
                    del predictions, trg
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
                
                # If chunk is full or this is the last batch, append to main lists and clear chunk
                if len(current_chunk['imgs']) >= chunk_size or batch_idx == len(test_loader) - 1:
                    imgs.extend(current_chunk['imgs'])
                    predicts.extend(current_chunk['preds'])
                    gt.extend(current_chunk['gts'])
                    
                    # Clear chunk
                    current_chunk = {'imgs': [], 'preds': [], 'gts': []}
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Unexpected error during testing: {e}")
            # Save what we have so far
            if current_chunk['imgs']:
                imgs.extend(current_chunk['imgs'])
                predicts.extend(current_chunk['preds'])
                gt.extend(current_chunk['gts'])
    
    return predicts, gt, imgs

def calculate_cer(pred_text, target_text):
    """Calculate Character Error Rate using Levenshtein distance."""
    if len(target_text) == 0:
        return 0 if len(pred_text) == 0 else 1
        
    matrix = [[0 for _ in range(len(pred_text) + 1)] 
              for _ in range(len(target_text) + 1)]
    
    for i in range(len(target_text) + 1):
        matrix[i][0] = i
    for j in range(len(pred_text) + 1):
        matrix[0][j] = j
        
    for i in range(1, len(target_text) + 1):
        for j in range(1, len(pred_text) + 1):
            if target_text[i-1] == pred_text[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1] + 1,    # substitution
                                 matrix[i][j-1] + 1,         # insertion
                                 matrix[i-1][j] + 1)         # deletion
                
    return matrix[len(target_text)][len(pred_text)] / len(target_text)

def calculate_wer(pred_text, target_text):
    """Calculate Word Error Rate using word-level Levenshtein distance."""
    pred_words = pred_text.split()
    target_words = target_text.split()
    
    if len(target_words) == 0:
        return 0 if len(pred_words) == 0 else 1
        
    matrix = [[0 for _ in range(len(pred_words) + 1)] 
              for _ in range(len(target_words) + 1)]
    
    for i in range(len(target_words) + 1):
        matrix[i][0] = i
    for j in range(len(pred_words) + 1):
        matrix[0][j] = j
        
    for i in range(1, len(target_words) + 1):
        for j in range(1, len(pred_words) + 1):
            if target_words[i-1] == pred_words[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1] + 1,    # substitution
                                 matrix[i][j-1] + 1,         # insertion
                                 matrix[i-1][j] + 1)         # deletion
                
    return matrix[len(target_words)][len(pred_words)] / len(target_words)
