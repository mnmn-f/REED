"""Evaluate an ordered KS50-C stream. All paths are relative to the caller."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from reed.runtime import REED


def load_model(path, device):
    from models.cav_mae import CAVMAEFT
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key,value in obj.get("state_dict",obj).items()}
    model = CAVMAEFT(label_dim=50)
    for projection in ("q","k","v"):
        for suffix in ("weight","bias"):
            state.pop(f"blocks_u.0.attn.{projection}.{suffix}",None)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data","checkpoint"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--benchmark", "--dasp", dest="benchmark", type=Path, required=True)
    p.add_argument("--scenario",choices=["S1","S2","S3"],required=True)
    p.add_argument("--protocol",choices=["continual","episodic"],required=True)
    p.add_argument("--gain-max",type=float,required=True)
    p.add_argument("--severity",type=int,choices=range(1,6),default=5)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.gain_max<=0: p.error("--gain-max must be positive")
    root = args.benchmark.resolve()
    data = args.data.resolve()
    checkpoint = args.checkpoint.resolve()
    assets = Path(__file__).resolve().parent/"assets"
    sys.path.insert(0,str(root))
    os.chdir(root)  # Official loader resolves class and test manifests from this directory.
    from datasets.data_loading import get_test_loader
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    model = load_model(checkpoint,args.device)
    config = yaml.safe_load((root/"cfgs/ks50_c/dasp.yaml").read_text())
    conditions = config["CORRUPTION"]["DOMAIN_SEQUENCE"]
    modality = {"S2":"video","S3":"audio"}.get(args.scenario)
    conditions = [c for c in conditions if modality is None or modality in c]
    state = None
    accuracies = []
    for condition in conditions:
        if state is None or args.protocol=="episodic":
            state = REED(model,assets,args.gain_max,beta=.9)
        loader = get_test_loader(args.protocol,"dasp","ks50",str(data),condition,conditions,
                                 args.severity,-1,1,64,shuffle=False,workers=0)
        hits = count = 0
        for audio,video,labels in loader:
            prob = state.predict_and_update(audio.to(args.device),video.to(args.device))
            hits += int((prob.argmax(1).cpu()==labels).sum())
            count += len(labels)
        accuracies.append(100*hits/count)
        print(json.dumps(dict(condition=condition,count=count,accuracy=accuracies[-1])),flush=True)
    print(json.dumps(dict(mean_accuracy=float(np.mean(accuracies)))))


if __name__=="__main__":
    main()
