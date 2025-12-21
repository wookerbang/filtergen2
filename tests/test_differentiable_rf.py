import unittest

import numpy as np
import torch

from src.data.circuits import abcd_to_sparams, components_to_abcd
from src.data.schema import ComponentSpec
from src.physics.differentiable_rf import DynamicCircuitAssembler, InferenceTimeOptimizer


class DifferentiableRFTests(unittest.TestCase):
    def _ladder_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("L", "series", 4.7e-9, None, "in", "n1"),
            ComponentSpec("C", "shunt", 1.0e-12, None, "n1", "gnd"),
            ComponentSpec("L", "series", 8.2e-9, None, "n1", "n2"),
            ComponentSpec("C", "shunt", 2.2e-12, None, "n2", "gnd"),
            ComponentSpec("L", "series", 3.3e-9, None, "n2", "out"),
        ]

    def test_torch_matches_numpy_abcd(self):
        comps = self._ladder_components()
        z0 = 50.0
        freq_hz = np.logspace(6, 9, 128)
        A, B, C, D = components_to_abcd(comps, freq_hz, z0)
        s21_db_np, _ = abcd_to_sparams(A, B, C, D, z0)

        assembler = DynamicCircuitAssembler(z0=z0)
        circuit, _ = assembler.assemble(comps, trainable=False, eps=1e-18, dtype=torch.float64)
        s21_db_t = circuit(torch.tensor(freq_hz, dtype=torch.float64), output="s21_db").detach().cpu().numpy()
        np.testing.assert_allclose(s21_db_t, s21_db_np, rtol=0, atol=1e-9)

    def test_gradients_flow_through_values(self):
        comps = self._ladder_components()
        z0 = 50.0
        freq_hz = torch.logspace(6, 9, 64, dtype=torch.float64)
        assembler = DynamicCircuitAssembler(z0=z0)
        circuit, _ = assembler.assemble(comps, trainable=False, eps=1e-18, dtype=torch.float64)

        values = torch.tensor([c.value_si for c in comps], dtype=torch.float64, requires_grad=True)
        pred = circuit(freq_hz, values=values, output="s21_db")
        loss = torch.mean(pred**2)
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all().item())

    def test_batch_values_supported(self):
        comps = self._ladder_components()
        z0 = 50.0
        freq_hz = torch.logspace(6, 9, 32, dtype=torch.float64)
        assembler = DynamicCircuitAssembler(z0=z0)
        circuit, _ = assembler.assemble(comps, trainable=False, eps=1e-18, dtype=torch.float64)

        base = torch.tensor([c.value_si for c in comps], dtype=torch.float64)
        values = torch.stack([base, base * 1.1], dim=0)
        out = circuit(freq_hz, values=values, output="s21_db")
        self.assertEqual(tuple(out.shape), (2, int(freq_hz.numel())))

    def test_refinement_reduces_loss_and_keeps_positive(self):
        z0 = 50.0
        freq_hz = np.logspace(6, 9, 64)
        true_comps = self._ladder_components()

        assembler = DynamicCircuitAssembler(z0=z0)
        true_circuit, _ = assembler.assemble(true_comps, trainable=False, eps=1e-18, dtype=torch.float64)
        target = true_circuit(torch.tensor(freq_hz, dtype=torch.float64), output="s21_db").detach()

        init_comps = [
            ComponentSpec(
                ctype=c.ctype,
                role=c.role,
                value_si=float(c.value_si) * 1.5,
                std_label=c.std_label,
                node1=c.node1,
                node2=c.node2,
            )
            for c in true_comps
        ]

        opt = InferenceTimeOptimizer(z0=z0)
        res = opt.refine(
            init_comps,
            freq_hz=freq_hz,
            target_s21_db=target,
            steps=60,
            lr=1e-1,
            max_ratio=3.0,
            optimizer="adam",
        )

        self.assertEqual(len(res.loss_history), 60)
        self.assertLess(res.final_loss, res.initial_loss)
        self.assertTrue(all(c.value_si > 0 for c in res.refined_components))


if __name__ == "__main__":
    unittest.main()

