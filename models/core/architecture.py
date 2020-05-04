from tensorflow.keras.models import Model

from models.core.blocks import InputBlock, ConvReluMap

class NetArchitecture:
    def __init__(self, arch_definition, outputs, input_shapes, output_sizes):
        self.arch_definition = arch_definition
        self.outputs = outputs
        self.nodes = self.__get_nodes()
        self.input_shapes = input_shapes
        self.output_sizes = output_sizes

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
        additional_inputs_tensors = {}
       
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

            # Gathering additional inputs
            if isinstance(node_to_process, ConvReluMap):
                if node_to_process.map_mask is not None:
                    additional_inputs_tensors[node_to_process.map_mask] = node_to_process.additional_input

            # Settling the states
            unresolved_nodes.remove(node_to_process.name)
            resolved_nodes[node_to_process.name] = node_output

        # create model
        additional_inputs = list(additional_inputs_tensors.keys())
        additional_inputs.sort()
        inputs = [self.get_node_by_name("input").input]
        for ai in additional_inputs:
            inputs.append(additional_inputs_tensors[ai])
        outputs = [resolved_nodes[o] for o in self.outputs]

        return Model(inputs=inputs, outputs = outputs)