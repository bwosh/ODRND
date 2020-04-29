from tensorflow.keras.models import Model

from models.core.blocks import InputBlock

class NetArchitecture:
    def __init__(self, arch_definition, outputs):
        self.arch_definition = arch_definition
        self.outputs = outputs
        self.nodes = self.__get_nodes()

    def __get_nodes(self)->dict:
        nodes = {}
        for node in self.arch_definition:
            if node.name in nodes.keys():
                raise Exception("Duplicated layer name.")

            if node.name == "input" and not isinstance(node, InputBlock):
                raise Exception("Input name badly used (reserver for inputs).")

            if node.name != "input" and isinstance(node, InputBlock):
                raise Exception("Input name is bad.")

            if node.name != "input" and len(node.inputs)==0:
                raise Exception(f"No inputs defined for: {node.name}")

            nodes[node.name] = node

        if "input" not in nodes.keys():
            raise Exception("No input found.")

        if len(self.outputs)==0:
            raise Exception("No outputs defined.")
        else:
            for o in self.outputs:
                if o not in nodes.keys():
                    raise Exception(f"Output node {o} not found")

        return nodes

    def get_node_by_name(self, name):
        return self.nodes[name]
        
    def to_model(self):
        
        input = self.get_node_by_name("input").forward(None)
        
        resolved_nodes = {"input":input}
        unresolved_nodes = [k for k in self.nodes if k!="input"]
       
        while(len(unresolved_nodes)>0):
            # Find next node with resolved inputs
            node_to_process = None
            for n in unresolved_nodes:
                can_process = True
                n = self.get_node_by_name(n)

                for dependency in n.inputs:
                    if dependency in unresolved_nodes:
                        can_process = False
                        break
                if can_process:
                    node_to_process = n
                    break

            # No node found yet there are unresolved
            if node_to_process is None:
                raise("Cannot resolve model")

            # Gathering dependencies
            inputs = []
            for dependency in node_to_process.inputs:
                inputs.append(resolved_nodes[dependency])

            # Forwarding
            node_output = node_to_process.forward(inputs)
            unresolved_nodes.remove(node_to_process.name)
            resolved_nodes[node_to_process.name] = node_output

        # create model
        intput = self.get_node_by_name("input").input
        outputs = [resolved_nodes[o] for o in self.outputs]

        return Model(inputs=input, outputs = outputs)