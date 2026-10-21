"""Small temporal evidence model. All arguments are measured sensor inputs."""
import torch
from torch import nn
import torch.nn.functional as F


class EvidenceModel(nn.Module):
    def __init__(self, feature_dim=1060, proprio_dim=128):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3,24,3,padding=1),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,32,3,padding=1),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,48,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((3,3)),
            nn.Flatten(),nn.Linear(432,64),nn.SiLU())
        self.evidence = nn.GRU(69,64,batch_first=True)
        self.symbol = nn.Linear(64,7)
        self.visibility = nn.Linear(64,1)
        self.sensor = nn.Sequential(nn.Linear(feature_dim+proprio_dim,192),nn.SiLU())
        self.temporal = nn.GRU(192,128,batch_first=True)
        self.pose = nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,6))
        self.ready = nn.Linear(128,6)
        self.view = nn.Sequential(nn.Linear(128+6*64,192),nn.SiLU(),nn.Linear(192,42))
        self.register_buffer("sensor_mean",torch.zeros(feature_dim+proprio_dim))
        self.register_buffer("sensor_std",torch.ones(feature_dim+proprio_dim))

    def forward(self, patches, geometry, features, proprio):
        # patches: B,T,6,3,48,48. Temporal RGB evidence never sees layout.
        b,t = patches.shape[:2]
        enc = self.patch(patches.reshape(-1,3,48,48)).reshape(b,t,6,64)
        evidence = torch.cat((enc,geometry),-1).permute(0,2,1,3).reshape(b*6,t,69)
        e = self.evidence(evidence)[0][:,-1].reshape(b,6,64)
        sensors = torch.cat((features,proprio),-1)
        sensors = ((sensors-self.sensor_mean)/self.sensor_std).clamp(-8,8)
        z = self.temporal(self.sensor(sensors))[0][:,-1]
        view = self.view(torch.cat((z,e.flatten(1)),1)).reshape(b,6,7)
        return dict(symbol=self.symbol(e),visible=self.visibility(e).squeeze(-1),
                    log_error=self.pose(z),ready=self.ready(z),
                    future_visible=view[...,:6],survival=view[...,6],latent=z)


def load_model(path, device):
    pack = torch.load(path,map_location=device,weights_only=True)
    model = EvidenceModel().to(device).eval()
    model.load_state_dict(pack["model"])
    model.requires_grad_(False)
    return model, pack
