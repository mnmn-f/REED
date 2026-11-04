import torch
def radial_layout(height,width,device):
 y=torch.arange(height,device=device);x=torch.arange(width,device=device)
 y=torch.minimum(y,height-y);x=torch.minimum(x,width-x)
 bins=(y[:,None].double().square()+x[None,:].double().square()).sqrt().round().long()
 counts=torch.bincount(bins.flatten()).double()
 return bins,counts

def radial_power(spectrum,bins,counts):
 power=spectrum.real.square()+spectrum.imag.square()
 out=torch.zeros((*power.shape[:2],len(counts)),device=power.device,dtype=torch.float64)
 out.scatter_add_(2,bins.flatten().expand(*power.shape[:2],-1),power.flatten(2))
 return out/counts.clamp_min(1)
def correct(rgb, reference, gain_max):
    """RGB float [B,3,H,W] in [0,1], before ImageNet normalization."""
    x = rgb.double()
    mean = x.mean((-2,-1), keepdim=True)
    spectrum = torch.fft.fft2(x-mean, norm='ortho')
    bins, counts = radial_layout(*x.shape[-2:], x.device)
    power = radial_power(spectrum,bins,counts).mean(0)
    gain = (reference.to(power)/(power+1e-12)).sqrt().clamp(max=gain_max)
    gain[:,0] = 1
    return (torch.fft.ifft2(spectrum*gain[:,bins],norm='ortho').real+mean).clamp(0,1).to(rgb)
