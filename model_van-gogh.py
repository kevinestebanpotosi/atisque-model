import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# ==========================================
# 1. ARQUITECTURA DEL MODELO (nncase friendly)
# ==========================================
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 # Padding estándar para K230
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        return x + self.norm(self.conv2(self.conv1(x)))

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Extracción de características
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # 5 Bloques residuales
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # Reconstrucción de la imagen (ConvTranspose2d nativo para la KPU)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.InstanceNorm2d(32, affine=True)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = torch.relu(self.norm1(self.up1(x)))
        x = torch.relu(self.norm2(self.up2(x)))
        return self.conv_final(x)

# ==========================================
# 2. RED VGG PARA MEDIR LA PÉRDIDA (El Juez)
# ==========================================
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg_pretrained[x] for x in range(4)])  # relu1_2
        self.slice2 = nn.Sequential(*[vgg_pretrained[x] for x in range(4, 9)]) # relu2_2
        self.slice3 = nn.Sequential(*[vgg_pretrained[x] for x in range(9, 16)])# relu3_3
        self.slice4 = nn.Sequential(*[vgg_pretrained[x] for x in range(16, 23)])#relu4_3
        # No necesitamos entrenar la VGG
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

def calc_gram_matrix(y):
    b, c, h, w = y.size()
    features = y.view(b, c, h * w)
    gram = features.bmm(features.transpose(1, 2))
    return gram / (c * h * w)

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO Y EXPORTACIÓN
# ==========================================
def entrenar_y_exportar():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando usando: {dispositivo}")

    # Hiperparámetros
    TAMANO_IMAGEN = 224 # Tamaño estático exigido por nncase
    BATCH_SIZE = 4
    EPOCHS = 1
    CONTENT_WEIGHT = 1e5
    STYLE_WEIGHT = 1e10

    # Cargar Dataset de Contenido (Las fotos normales)
    transformacion = transforms.Compose([
        transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Asegúrate de tener imágenes en dataset/imagenes/
    train_dataset = ImageFolder("dataset", transform=transformacion)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Cargar y procesar la Imagen de Estilo (Van Gogh)
    estilo_img = Image.open("estilo/starry_night.jpg").convert('RGB')
    estilo_tensor = transformacion(estilo_img).unsqueeze(0).to(dispositivo)
    estilo_tensor = estilo_tensor.repeat(BATCH_SIZE, 1, 1, 1)

    # Iniciar Modelos
    red_generadora = TransformerNet().to(dispositivo)
    vgg = VGGFeatures().to(dispositivo)
    optimizer = optim.Adam(red_generadora.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # Pre-calcular el "Estilo" de Van Gogh para que sea rápido
    caracteristicas_estilo = vgg(estilo_tensor)
    gram_estilo = [calc_gram_matrix(y) for y in caracteristicas_estilo]

    print("Iniciando entrenamiento...")
    for epoch in range(EPOCHS):
        for batch_id, (x_contenido, _) in enumerate(train_loader):
            n_batch = len(x_contenido)
            x_contenido = x_contenido.to(dispositivo)

            optimizer.zero_grad()

            # 1. Pasar imagen por nuestra red (Generar arte)
            y_generado = red_generadora(x_contenido)

            # 2. Extraer características con VGG
            vgg_generado = vgg(y_generado)
            vgg_contenido = vgg(x_contenido)

            # 3. Calcular Pérdida de Contenido (que se siga viendo la imagen original)
            loss_contenido = CONTENT_WEIGHT * mse_loss(vgg_generado[1], vgg_contenido[1])

            # 4. Calcular Pérdida de Estilo (que se parezca a Van Gogh)
            loss_estilo = 0.0
            for ft_y, gm_s in zip(vgg_generado, gram_estilo):
                gm_y = calc_gram_matrix(ft_y)
                loss_estilo += mse_loss(gm_y, gm_s[:n_batch, :, :])
            loss_estilo *= STYLE_WEIGHT

            # 5. Ajustar pesos
            loss_total = loss_contenido + loss_estilo
            loss_total.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_id} | Loss: {loss_total.item():.2f}")

    print("Entrenamiento completado.")

    # ==========================================
    # 4. EXPORTAR A ONNX PARA LA CANMV K230
    # ==========================================
    print("Exportando modelo a ONNX...")
    red_generadora.eval().cpu()
    
    # Creamos un tensor vacío con el tamaño exacto de entrada
    entrada_simulada = torch.randn(1, 3, TAMANO_IMAGEN, TAMANO_IMAGEN)
    
    torch.onnx.export(
        red_generadora, 
        entrada_simulada, 
        "noche_estrellada_k230.onnx", 
        export_params=True,
        opset_version=11, # Versión 11 es la más segura y estable para nncase
        input_names=['input'], 
        output_names=['output']
    )
    print("¡Éxito! Archivo 'noche_estrellada_k230.onnx' generado correctamente.")

if __name__ == "__main__":
    entrenar_y_exportar()