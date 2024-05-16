import numpy as np

class WorkloadOptimizer:
    def __init__(self, dc_ids):
        self.dc_ids = dc_ids
        # Initialize the transfer matrix with zeros
        self.transfer_matrix = {from_dc: {to_dc: 0 for to_dc in dc_ids} for from_dc in dc_ids}

    
    def compute_adjusted_workload(self, dc_states):
        """
        Computes the optimal adjustments to the workload of each data center to minimize carbon emissions.
        
        :param dc_states: A dictionary with data center states, where each state includes:
                          - capacity (MWh)
                          - current workload
                          - external temperature
                          - energy carbon intensity
        :return: A dictionary with the recommended workload adjustments for each data center.
        """
        # Implement the optimization logic here.
        total_mwh = {
            dc_id: state['dc_capacity'] * state['curr_workload'] 
            for dc_id, state in dc_states.items()
            }
        
        # Sort data centers by their carbon intensity (from lowest to highest)
        sorted_dcs = sorted(dc_states.items(), key=lambda x: x[1]['ci'])
        
        # Reset the transfer matrix to zero for each computation
        for from_dc in self.dc_ids:
            for to_dc in self.dc_ids:
                self.transfer_matrix[from_dc][to_dc] = 0
                

        max_cap = 0.90 # To prevent to loss tasks when posponed
        for i in range(len(sorted_dcs)):
            receiver_id, receiver_state = sorted_dcs[i]
            receiver_capacity = receiver_state['dc_capacity'] * (max_cap - receiver_state['curr_workload'])
            
            for j in range(len(sorted_dcs) - 1, i, -1):
                sender_id, sender_state = sorted_dcs[j]
                sender_workload = total_mwh[sender_id]
                
                workload_to_move = min(sender_workload, receiver_capacity)
                receiver_capacity -= workload_to_move
                total_mwh[sender_id] -= workload_to_move
                total_mwh[receiver_id] += workload_to_move
                
                # Update the transfer matrix
                self.transfer_matrix[sender_id][receiver_id] += workload_to_move
                
                if receiver_capacity <= 1-max_cap:
                    break

        # Calculate net MWh added or removed for each data center
        net_mwh_changes = {
            dc_id: sum(row[dc_id] for row in self.transfer_matrix.values())
            - sum(self.transfer_matrix[dc_id].values())
            for dc_id in self.dc_ids
        }
        
        # Convert the net MWh change into workload percentage and adjust the original workload percentages
        new_workloads = {

            dc_id: (dc_states[dc_id]['curr_workload'] + net_mwh_changes[dc_id]) / dc_states[dc_id]['dc_capacity'] 
            if dc_states[dc_id]['dc_capacity'] > 0 else 0
            for dc_id in self.dc_ids
        }

        return new_workloads, self.transfer_matrix

    def compute_actions(self, obs: dict) -> dict:

        _, transfer_matrix = self.compute_adjusted_workload(obs)
        
        # Create a dictionary of actions from the transfer matrix
        actions = {}
        transfer_id = 1  # To keep track of each transfer action
        # Convert dict_keys to a list for indexing
        # self.dc_ids = list(obs.keys())
        for sender, receivers in transfer_matrix.items():
            for receiver, amount in receivers.items():
                if amount > 0:  # Only consider positive transfers
                    sender_index = self.dc_ids.index(receiver)
                    receiver_index = self.dc_ids.index(sender)
                    actions[f'transfer_{transfer_id}'] = {
                        'sender': sender_index,
                        'receiver': receiver_index,
                        'workload_to_move': np.array([amount], dtype=float)
                    }
                    transfer_id += 1

        return actions