import unittest

import torch

from model import BRAVE, LightweightTemporalTransformer


torch.set_num_threads(1)


class TestLightweightTemporalTransformer(unittest.TestCase):
    def test_shape_parameters_and_gradients(self):
        module = LightweightTemporalTransformer(
            dim=24,
            num_heads=3,
            num_frames=4,
            num_layers=2,
        )
        x = torch.randn(8, 9, 24, requires_grad=True)
        y = module(x, H=3, W=3)

        self.assertEqual(tuple(y.shape), (8, 9, 24))
        self.assertEqual(tuple(module.temporal_pos_embed.shape), (1, 4, 24))

        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(module.temporal_pos_embed.grad.abs().sum().item(), 0.0)
        self.assertGreater(module.context_proj.weight.grad.abs().sum().item(), 0.0)

    def test_temporal_position_encoding_makes_frame_order_observable(self):
        torch.manual_seed(7)
        module = LightweightTemporalTransformer(
            dim=24,
            num_heads=3,
            num_frames=4,
            num_layers=1,
        ).eval()
        with torch.no_grad():
            position = torch.arange(4 * 24, dtype=torch.float32).view(1, 4, 24) / 50.0
            module.temporal_pos_embed.copy_(position)

        x = torch.randn(1, 4, 9, 24)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)

        with torch.no_grad():
            y = module(x.view(4, 9, 24), H=3, W=3).view(1, 4, 9, 24)
            y_permuted = module(
                x[:, permutation].contiguous().view(4, 9, 24),
                H=3,
                W=3,
            ).view(1, 4, 9, 24)
            y_permuted = y_permuted[:, inverse]

        max_difference = (y - y_permuted).abs().max().item()
        self.assertGreater(max_difference, 1e-5)


class TestBRAVEArchitecture(unittest.TestCase):
    @staticmethod
    def build_small_model():
        return BRAVE(
            num_classes=2,
            num_frames=4,
            depths=(1, 1, 1, 1),
            window_size=4,
            drop_path_rate=0.0,
        )

    def test_four_temporal_stages_and_input_layouts(self):
        torch.manual_seed(11)
        model = self.build_small_model().eval()

        self.assertEqual(len(model.temporal_modules), 4)
        specs = [
            (module.dim, module.num_layers, module.num_heads)
            for module in model.temporal_modules
        ]
        self.assertEqual(
            specs,
            [(192, 1, 2), (384, 1, 3), (768, 2, 6), (768, 2, 6)],
        )

        calls = []
        handles = []

        def make_hook(index):
            def hook(module, inputs, output):
                calls.append((index, tuple(inputs[0].shape), tuple(output.shape)))
            return hook

        for index, module in enumerate(model.temporal_modules):
            handles.append(module.register_forward_hook(make_hook(index)))

        x_channels_first = torch.randn(1, 4, 3, 64, 64)
        with torch.no_grad():
            y_channels_first = model(x_channels_first)

        for handle in handles:
            handle.remove()

        self.assertEqual(tuple(y_channels_first.shape), (1, 2))
        self.assertEqual(
            calls,
            [
                (0, (4, 64, 192), (4, 64, 192)),
                (1, (4, 16, 384), (4, 16, 384)),
                (2, (4, 4, 768), (4, 4, 768)),
                (3, (4, 4, 768), (4, 4, 768)),
            ],
        )

        x_channels_last = x_channels_first.permute(0, 1, 3, 4, 2).contiguous()
        with torch.no_grad():
            y_channels_last = model(x_channels_last)
        torch.testing.assert_close(y_channels_first, y_channels_last)

    def test_stage4_temporal_module_receives_gradients(self):
        torch.manual_seed(13)
        module = LightweightTemporalTransformer(
            dim=768,
            num_heads=6,
            num_frames=4,
            num_layers=2,
        )
        x = torch.randn(4, 4, 768)
        loss = module(x, H=2, W=2).square().mean()
        loss.backward()

        self.assertGreater(module.temporal_pos_embed.grad.abs().sum().item(), 0.0)
        self.assertGreater(module.context_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            module.blocks[0]["attn"].in_proj_weight.grad.abs().sum().item(),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
