import torch
import logging
import os
from datetime import datetime
from utils.UNeXt import UNet

class TracingUNet(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up logger with datetime in filename
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f'tracing_unet_{dt_str}.txt')
        self.logger = logging.getLogger(f'TracingUNet_{dt_str}')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def forward(self, x, return_input_after_BN=False):
        latents = []

        self.logger.info("Input -> Prepended Layers")
        for i, cell in enumerate(self.prepended_layers):
            x = cell(x)
            layer_name = f"prepended_layers_{i}"
            self.logger.info(f"{layer_name} output shape: {x.shape}")

        for L, layer in enumerate(self.layers_encode):
            if L < len(self.layers_encode) - 1:
                Lx = x.clone()
                for j, block in enumerate(self.interlayer_cnn[L]):
                    Lx = block(Lx)
                    interlayer_name = f"interlayer_cnn_{L}_{j}"
                    self.logger.info(f"{interlayer_name} output shape: {Lx.shape}")
                latents.append(Lx)
                self.logger.info(f"Encoder block {L}: output -> latent_{L} shape: {Lx.shape}")
            for k, block in enumerate(layer):
                x = block(x)
                encode_layer_name = f"layers_encode_{L}_{k}"
                self.logger.info(f"{encode_layer_name} output shape: {x.shape}")

        for n, (layer, latent) in enumerate(zip(self.layers_decode, latents[::-1])):
            x = layer[0](x)
            decode_upsample_name = f"layers_decode_{n}_0"
            self.logger.info(f"{decode_upsample_name} (upsample) output shape: {x.shape}")
            concat_tensor = torch.cat([x, latent], axis=1)
            raw_concat_name = f"layers_decode_{n}_raw_concat"
            self.logger.info(f"{raw_concat_name} (raw concat) output shape: {concat_tensor.shape}")
            x = layer[1][0](concat_tensor)
            decode_concat_name = f"layers_decode_{n}_1_0"
            self.logger.info(f"{decode_concat_name} (concat) output shape: {x.shape}")
            for m, block in enumerate(layer[1][1:]):
                x = block(x)
                decode_resnet_name = f"layers_decode_{n}_1_{m+1}"
                self.logger.info(f"{decode_resnet_name} (resnet) output shape: {x.shape}")
            x = layer[2](x)
            decode_final_name = f"layers_decode_{n}_2"
            self.logger.info(f"{decode_final_name} (final conv) output shape: {x.shape}")

        for i, cell in enumerate(self.appended_layers):
            x = cell(x)
            appended_name = f"appended_layers_{i}"
            self.logger.info(f"{appended_name} output shape: {x.shape}")

        return x

    def register_logging_hooks(self):
        """
        Register forward hooks on all submodules to log their output shapes.
        """
        def hook_fn(module, input, output):
            # Only log for leaf modules (no children)
            if len(list(module.children())) == 0:
                # Try to get the module's name from the parent
                for name, mod in self.named_modules():
                    if mod is module:
                        self.logger.info(f"[HOOK] {name} output shape: {output.shape}")
                        break
        for name, module in self.named_modules():
            if name == "":
                continue  # skip the root module
            module.register_forward_hook(hook_fn)

# Usage example:
if __name__ == "__main__":
    # Load your model parameters as needed
    dummy_input = torch.randn(1, 1, 128, 128)  # Adjust shape as needed

    # Instantiate your model with the correct arguments
    # tracing_model = TracingUNet(...)

    # tracing_model.register_logging_hooks()  # <-- Register hooks before running forward
    # tracing_model.eval()
    # with torch.no_grad():
    #     output = tracing_model(dummy_input)