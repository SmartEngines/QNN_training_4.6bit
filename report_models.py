from torchinfo import summary
from cifar_models import CNN6, CNN7, CNN8, CNN9, CNN10

summary(CNN6().cpu(), (5, 3, 32, 32), device='cpu')
summary(CNN7().cpu(), (5, 3, 32, 32), device='cpu')
summary(CNN8().cpu(), (5, 3, 32, 32), device='cpu')
summary(CNN9().cpu(), (5, 3, 32, 32), device='cpu')
summary(CNN10().cpu(), (5, 3, 32, 32), device='cpu')
