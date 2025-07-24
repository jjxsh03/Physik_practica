import torch

# set of modelling classes

class LinearModel(torch.nn.Module):
    '''
    Linear fitting Model predicting optimal parameters for a predefined Epoch:
        fitting_data: x -> float
        device = gpu
    '''
    def __init__(self):
        super(LinearModel, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
    
    def forward(self, x) -> float:
        return self.a * x + self.b


class ExpModel(torch.nn.Module):
    '''
    Model for fitting of data with an exponential trend:
        training epochs: 300 -> 1000
    '''
    def __init__(self):
        super(ExpModel, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(()))
        self.B = torch.nn.Parameter(torch.tensor(0.1)) # gradiant descent updating 

    def forward(self, x) -> float:
        return self.A * torch.exp(self.B * x)

class CauchyModel(torch.nn.Module):
    '''
    General Fit for Lorentz/Cauchy Funtion
    '''
    def __init__(self):
        super(CauchyModel, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(()))
        self.x_0 = torch.nn.Parameter(torch.randn(()))
        self.gamma = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.A / (pow((x - self.x_0), 2) + pow(self.gamma, 2))

    def forward_normalized(self, x, *args):
        return self.gamma / (torch.pi * (pow((x - self.x_0), 2) + pow(self.gamma, 2)))

class SigmoidModel(torch.nn.Module):
    '''
    Fit Model for Hysteresis for both directions
    '''
    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(()))
        self.k = torch.nn.Parameter(torch.randn(()))
        self.x_c = torch.nn.Parameter(torch.randn(()))
    
    def forward(self, x, direction):
        if direction == 'UP':
            return self.A * torch.tanh(self.k * (x + self.x_c))
        elif direction == 'DOWN':
            return self.A * torch.tanh(self.k * (x - self.x_c))
        else:
            raise ValueError('Unallowed or No direction defined! \n Please choose UP or DOWN as directional Input')


# Defining modelling workflow 
class ModelData:
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, x, y, times, device='cpu'):
        x_train = x
        y_train = y
        epochs = times
        self.model.to(device)

        # Determine print frequency
        if epochs >= 1000:
            print_every = 100
        elif epochs >= 500:
            print_every = 50
        else:
            print_every = 25
        
        # Training of fit model
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            prediction = self.model(x_train)
            loss = self.loss_fn(prediction, y_train)
            loss.backward()
            self.optimizer.step()

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
