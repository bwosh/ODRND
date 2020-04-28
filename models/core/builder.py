from models.core.blocks import InputBlock
from models.core.architecture import NetArchitecture

def get_nodes(arch: NetArchitecture)->dict:
    nodes = {}
    for node in arch:
        if node.name in nodes.keys():
            raise Exception("Duplicated layer name.")

        if node.name == "input" and not isinstance(node, InputBlock):
            raise Exception("Input name badly used (reserver for inputs).")

        if node.name != "input" and isinstance(node, InputBlock):
            raise Exception("Input name is bad.")

        nodes[node.name] = node

    if "input" not in nodes.keys():
        raise Exception("No input found.")

    if len(arch.outputs)==0:
        raise Exception("No outputs defined.")
    else:
        for o in arch.outputs:
            if o not in nodes.keys():
                raise Exception(f"Output node {o} not found")


    # TODO no output , input must have input name
    return nodes
