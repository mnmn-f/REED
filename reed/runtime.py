from pathlib import Path
import numpy as np
import torch
from .adaptation import OnlineAdapters
from .diagnosis import JointRouter, GlobalChangeClock
from .spectrum import correct


class REED:
    def __init__(self, model, assets, gain_max, beta):
        self.model = model.eval().requires_grad_(False)
        self.device = next(model.parameters()).device
        self.assets = Path(assets)
        self.gain_max = gain_max
        with np.load(self.assets/'reference.npz') as z:
            self.power = torch.from_numpy(z['power']).to(self.device)
            self.adapters = OnlineAdapters(torch.from_numpy(z['mean']),torch.from_numpy(z['aps']),beta).to(self.device)
        self.optimizer = torch.optim.Adam(self.adapters.parameters(),lr=1e-4)
        self.routers = {}
        self.clock = None

    def predict_and_update(self, audio, video):
        """Inputs are official normalized filterbanks and normalized RGB frames."""
        batch = len(audio)
        if batch not in self.routers:
            router = JointRouter(self.assets/f'router_b{batch}.npz')
            router.root = self.assets
            self.routers[batch] = router
        router = self.routers[batch]
        if self.clock is None:
            self.clock = GlobalChangeClock(router)
        with torch.no_grad():
            raw = self.model.forward_encoder(audio,video)
            features = []
            for x in raw:
                for block in self.model.blocks_u:
                    x = block(x)
                features.append(self.model.norm(x).mean(1).cpu().numpy())
            diagnosis = router.predict(*features)
            changed = self.clock.update(diagnosis,features)
            processed = list(raw)
            if diagnosis['corrupted'][1]:
                mean = video.new_tensor([.485,.456,.406])[None,:,None,None]
                std = video.new_tensor([.229,.224,.225])[None,:,None,None]
                rgb = correct(video*std+mean,self.power,self.gain_max)
                processed[1] = self.model.forward_encoder(audio,(rgb-mean)/std)[1]
        return self.adapters.step(self.model.forward_decoder,raw,processed,diagnosis['corrupted'],changed,self.optimizer)
