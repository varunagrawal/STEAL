from steal.rmpflow import DimSelectorTaskMap, RmpTreeNode
from steal.rmpflow.controllers import DampingMomemtumController


def create_rmp_tree(cspace_dim, rmp_order):
    """
    Create initial RMP tree with the root node and damping nodes for each joint.
    """
    # ------------------------------------------------------
    print('Setting up tree')
    root = RmpTreeNode(n_dim=cspace_dim,
                       name="cspace_root",
                       order=rmp_order,
                       return_natural=True)
    root.eval()

    # --------------------------------
    print("Adding damping to each joint")
    joint_damping_gain = 1e-4
    for i in range(cspace_dim):
        joint_task_map = DimSelectorTaskMap(n_inputs=cspace_dim,
                                            selected_dims=i)
        joint_node = root.add_task_space(joint_task_map, name="joint" + str(i))
        damping_rmp = DampingMomemtumController(
            damping_gain=joint_damping_gain)
        joint_node.add_rmp(damping_rmp)

    return root
