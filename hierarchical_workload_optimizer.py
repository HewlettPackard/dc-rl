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
        total_mwh = {dc_id: state[0] * state[1] for dc_id, state in dc_states.items()}
        
        # Sort data centers by their carbon intensity (from lowest to highest)
        sorted_dcs = sorted(dc_states.items(), key=lambda x: x[1][3])
        
        # Reset the transfer matrix to zero for each computation
        for from_dc in self.dc_ids:
            for to_dc in self.dc_ids:
                self.transfer_matrix[from_dc][to_dc] = 0
                

        max_cap = 0.90 # To prevent to loss tasks when posponed
        for i in range(len(sorted_dcs)):
            receiver_id, receiver_state = sorted_dcs[i]
            receiver_capacity = receiver_state[0] * (max_cap - receiver_state[1])
            
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
        net_mwh_changes = {dc_id: sum(row[dc_id] for row in self.transfer_matrix.values()) - sum(self.transfer_matrix[dc_id].values()) for dc_id in self.dc_ids}

        # Convert the net MWh change into workload percentage and adjust the original workload percentages
        new_workloads = {
            dc_id: dc_states[dc_id][1] + net_mwh_changes[dc_id] / dc_states[dc_id][0] if dc_states[dc_id][0] > 0 else 0
            for dc_id in self.dc_ids
        }

        return new_workloads, self.transfer_matrix

