import os
import numpy as np
import torch
from genotypes import PRIMITIVES
from config_utils.decode_args import obtain_decode_args


class CamoNASDecoder(object):
    def __init__(self, args):
        self.args = args

        assert args.resume is not None, "Must specify --resume checkpoint"
        assert os.path.isfile(args.resume), f"Checkpoint not found: {args.resume}"

        ckpt = torch.load(args.resume, map_location="cpu")
        self.alphas = ckpt["state_dict"]["alpha"]

    def decode(self):
        genotype = []
        for i, a in enumerate(self.alphas):
            w = torch.softmax(a, dim=-1)
            idx = torch.argmax(w).item()
            primitive = PRIMITIVES[idx]
            genotype.append(primitive)
        return genotype

    def save(self, genotype):
        save_dir = os.path.dirname(self.args.resume)
        save_path = os.path.join(save_dir, "genotype.npy")
        np.save(save_path, genotype)
        print(f"Genotype saved to: {save_path}")


def main():
    args = obtain_decode_args()
    decoder = CamoNASDecoder(args)
    genotype = decoder.decode()

    print("=== CamoNAS Final Cell ===")
    for i, op in enumerate(genotype):
        print(f" Step {i}: {op}")

    decoder.save(genotype)


if __name__ == "__main__":
    main()
