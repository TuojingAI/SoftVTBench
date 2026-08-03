import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is installed in the Isaac environment")
class VectorStateContractTest(unittest.TestCase):
    def test_states_to_world_offsets_roots_and_each_deformable_node(self):
        import torch

        from softvtbench.evaluation.rollout_vec import _states_to_world

        state = {
            "articulation": {
                "robot": {
                    "root_pose": torch.tensor(
                        [
                            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                            [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                }
            },
            "rigid_object": {
                "target": {
                    "root_pose": torch.tensor(
                        [
                            [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                            [3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                }
            },
            "deformable_object": {
                "soft": {
                    "nodal_position": torch.tensor(
                        [
                            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                            [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                        ]
                    )
                }
            },
        }
        origins = torch.tensor([[0.0, 10.0, 0.0], [20.0, 0.0, 0.0]])

        world = _states_to_world(state, origins)

        self.assertTrue(
            torch.equal(
                world["articulation"]["robot"]["root_pose"][:, :3],
                torch.tensor([[1.0, 12.0, 3.0], [24.0, 5.0, 6.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                world["rigid_object"]["target"]["root_pose"][:, :3],
                torch.tensor([[0.0, 11.0, 2.0], [23.0, 4.0, 5.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                world["deformable_object"]["soft"]["nodal_position"],
                torch.tensor(
                    [
                        [[0.0, 10.0, 0.0], [1.0, 11.0, 1.0]],
                        [[22.0, 2.0, 2.0], [23.0, 3.0, 3.0]],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                state["deformable_object"]["soft"]["nodal_position"],
                torch.tensor(
                    [
                        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                        [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
