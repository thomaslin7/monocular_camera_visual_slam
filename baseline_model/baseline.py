#imports
import torch
import torch.nn as nn
import torchvision.models as models

#pretrained model - import, extract features, freeze weights (for transfer learning)
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)  # Updated from deprecated pretrained=True
encoder = vgg16.features 
for param in encoder.parameters():
    param.requires_grad = False

#decoder architecture (symetrical to vgg16 encoder architecture with transposed convolutions)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            #final output channel = 1 for depth map (may need to change)
        )
        
    def forward(self, x):
        return self.decoder(x)

#combined encoder and decoder (parameters are encoder and a Decoder instance)
class fullBaseline(nn.Module):
    def __init__(self, encoder, decoder):
        super(fullBaseline, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#model training function (parameters are the training and validation loaders, and the number of epochs)
def train_model(train_loader, val_loader, num_epochs):

    #seed 
    torch.manual_seed(42)
    
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #model instance
    decoder = Decoder()
    model = fullBaseline(encoder, decoder).to(device)

    #optimizer and loss function
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #training loop
    outputLog = []
    for epoch in range(num_epochs):

        #training mode for BatchNorm from VGG16
        model.train()
        train_loss = 0
        num_batches = 0

        for images, depths in train_loader:
            images = images.to(device)
            depths = depths.to(device)
            
            # Zero gradients before forward pass
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, depths)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1


        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0

        #eval mode for validation
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for images, depths in val_loader:
                images = images.to(device)
                depths = depths.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, depths)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
        outputLog.append((epoch, images.cpu(), avg_val_loss, outputs.cpu()))
    
    return outputLog, model  # Return both training log and the trained model


