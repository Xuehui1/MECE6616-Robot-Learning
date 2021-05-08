from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from train_dynamics import Net


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = Net()
        self.model_loaded = True
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            self.model.eval()
            #X = np.concatenate((state, action))
            X = np.concatenate((state, action)).T
            new_state = self.model(torch.FloatTensor(X)).detach().numpy()
            #return new_state
            return new_state.T
            # ---
        else:
            return state
