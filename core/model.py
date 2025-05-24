import torch
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Module, ModuleList, ReLU, BatchNorm2d, Sequential


class Block (Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, 3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, 3, padding=1),
            BatchNorm2d(out_channels),
            ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class UNet (Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # todo: zu viele Zahlen, zu viel Wiederholungen. Sollte anders gelöst werden! siehe unten
        self.encoder = ModuleList([Block(in_channels, 64),
                                   Block(64, 128),
                                   Block(128, 256),
                                   Block(256, 512)])
        # self.encoder = ModuleList([Block(i,o) for i,o in
        #                            [(in_channels, 64),
        #                             (64,128),
        #                             (128,256),
        #                             (256,512)]])
        self.pool = MaxPool2d(2)
        self.bottleneck = Block(512, 1024)
        self.decoder = ModuleList([Block(1024, 512),
                                   Block(512, 256),
                                   Block(256, 128),
                                   Block(128, 64)])
        self.up_samples = ModuleList([ConvTranspose2d(1024,512, 2, 2),
                                   ConvTranspose2d(512,256, 2, 2),
                                   ConvTranspose2d(256,128, 2, 2),
                                   ConvTranspose2d(128,64, 2, 2)])
        self.head = Conv2d(64, out_channels, 1)

    def crop(self, encoder_block_output, x):
        """
        :param encoder_block_output: output of the encoder used for skip connections, to crop
        :param x: input that is not to crop but defines the size
        :return: encoder output cropped to size of x
        """
        (_, _, x_height, x_width) = x.shape
        encoder_block_output = encoder_block_output[:,:,:x_height,:x_width]
        # encoder_block_output = torchvision.transforms.v2.functional.center_crop(encoder_block_output, [x_height, x_width])
        # todo: CenterCrop statt slicing
        return encoder_block_output

    def forward(self, x):
        encoder_outputs = []  # for skip connections to decoder path

        # Encoder:
        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder:
        for i, block in enumerate(self.decoder):
            x = self.up_samples[i](x)
            skip_connection = encoder_outputs.pop() # take the last
            skip_connection = self.crop(skip_connection, x)  # ensure dimensions fit
            x = torch.cat([x, skip_connection], dim=1)
            x = block(x)

        x = self.head(x)
        return x

    def get_parameters_count(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == "__main__":
    unet = UNet()
    print(f"Number of parameters: {unet.get_parameters_count()}")

    # dry run:
    dummy_x = torch.rand(size=(3,256, 256))
    print(f"Shape of dummy input: {dummy_x.shape}")
    dummy_y = unet(dummy_x.unsqueeze(0)) # fake a batch channel
    #make_dot(dummy_y.mean(), params=dict(unet.named_parameters())).render("unet_torchviz", format="png")
    print(f"Shape of dummy output: {dummy_y.shape}")