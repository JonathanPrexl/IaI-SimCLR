import re
import torch
import numpy as np
from numpy import dtype
import torchvision


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def color_jitter_multispectral(tensor):
    
    s = 1

    jitter_func = torchvision.transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

    # assuming channel first
    C, W, H = tensor.shape

    if min([C,W,H]) != C:
        raise ValueError("More channels then with or high pixels seems suspicious.")

    # iterating over channels
    # and overwriting the band with a 
    # transformed version
    for cidx in range(C):
        tensor[cidx] = jitter_func(tensor[cidx].unsqueeze(axis=0))

    return tensor

def greyscale_multichannel(x):

    """ often there are weights on the bands
    here we just apply the greyscale option to the first
    tree bands and ignoring more channel inputs
    """

    # assuming channel first
    C, W, H = x.shape

    x = x[:3]
    B,G,R = x # the order in S2 images is B G R 
    x = torch.stack([R,G,B],axis=0)

    grey_2d = torchvision.transforms.RandomGrayscale(p=1)(x)[:1]
    grey_3d = grey_2d.repeat(C, 1, 1)

    return grey_3d

class concatGreyAndJitter:
    """
    combines grey and jitter so that we dont make
    grey images from the jittered one... we dont want to apply those
    ops twice.
    """

    def __init__(self,grey_prob: float, jitter_prob: float) -> None:
        self.grey_prob = grey_prob
        self.jitter_prob = jitter_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        
        rn = torch.rand(1)

        if rn <= self.jitter_prob:
            return color_jitter_multispectral(x)
        elif self.jitter_prob < rn <= self.jitter_prob+self.grey_prob:
            return greyscale_multichannel(x)
        else:
            return x
        

class Dummy:
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        return x, torch.empty((2,2,2)), torch.empty((2,2,2))
        

class TransformsPostivePair:
    
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, imgsize: int,
                       crop_active: bool,
                       crop_prob: float,
                       crop_scale: tuple,
                       flip_active: bool,
                       flip_prob: bool,
                       grey_active: bool,
                       grey_prob: float,
                       color_jitter_active: bool,
                       color_jitter_prob: float,
                       blur_active: bool,
                       blur_prob: float,
                       ) -> None:

        self.imgsize = imgsize

        # if we have both, greyscale and color jtter.. make sure combined
        # probabily is smaller or equal to one
        if grey_active and color_jitter_active:
            assert grey_prob + color_jitter_prob <= 1, "otherwise its not working"

        trasformationlist = []

        # works out of the box
        if crop_active:
            crop = torchvision.transforms.RandomResizedCrop(size=imgsize,scale=tuple(crop_scale))
            trasformationlist.append(torchvision.transforms.RandomApply([crop], p=crop_prob))

        if flip_active:
            flip = torchvision.transforms.RandomHorizontalFlip(p=1)
            trasformationlist.append(torchvision.transforms.RandomApply([flip], p=flip_prob))

        if grey_active and color_jitter_active:
            jitter_and_grey = concatGreyAndJitter(grey_prob=grey_prob,
                                                  jitter_prob=color_jitter_prob)
            # dont have to be randomly applyied since we took care of that
            trasformationlist.append(jitter_and_grey)

        else:
            if grey_active:
                grey = greyscale_multichannel
                trasformationlist.append(torchvision.transforms.RandomApply([grey], p=grey_prob))
            if color_jitter_active:
                color_jitter = color_jitter_multispectral
                trasformationlist.append(torchvision.transforms.RandomApply([color_jitter], p=color_jitter_prob))

        if blur_active:
            ten_percent_of_img = 10 * imgsize/100
            blur = torchvision.transforms.GaussianBlur(kernel_size=round_up_to_odd(ten_percent_of_img))
            trasformationlist.append(torchvision.transforms.RandomApply([blur], p=blur_prob))

        self.transformsComposed = torchvision.transforms.Compose(
               trasformationlist
            )

    def __call__(self, x):
        assert x.shape[-1] == self.imgsize
        return self.transformsComposed(x.clone())


if __name__ == "__main__":

    import numpy as np
    import torch
    import torchvision


    nC = 4
    x  = torch.rand(nC,256,256)
    transformer = TransformsPostivePair(imgsize=256,
                                        crop_active=True,
                                        crop_prob=.5,
                                        crop_scale=(.5,1),
                                        flip_active=True,
                                        flip_prob=.3,
                                        grey_active=True,
                                        grey_prob=.1,
                                        color_jitter_active=True,
                                        color_jitter_prob=.8,
                                        blur_active=True,
                                        blur_prob=.2,)
    a,b,c = transformer(x)