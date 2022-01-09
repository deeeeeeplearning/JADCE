import argparse

class Options():
    
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self,parser):
        parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
        parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
        parser.add_argument("--N", type=int, default=256, help="number of devices")
        parser.add_argument("--pro", type=float, default=0.1, help="probaility of activity")
        parser.add_argument("--M", type=int, default=8, help="size of antenna")
        parser.add_argument("--channels", type=int, default=2, help="number of image channels")
        parser.add_argument("--L", type=int, default=128, help="size of sequense")
        parser.add_argument("--weight1", type=float, default=1,help="weight of discriminator loss for generator")
        parser.add_argument("--name", type=str, default='Unet_gan', help="model name")
        parser.add_argument("--device_id", type=int, default='0', help="device_id")
        parser.add_argument("--loop", type=int, default='12', help="the number of Unet-block or (LISTA)-block")
        parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
        parser.add_argument("--SNR", type=float, default='30', help="var of noise")
        parser.add_argument("--wgan", action='store_true',help="WGAN")
        parser.add_argument("--trained_P", action='store_true',help='whether design pilot matrix')
        self.initialized = True
        return parser
        
    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        print(opt)
        self.opt = opt
        return self.opt
        
        

        
        
