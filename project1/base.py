import abc


class RobotPolicy(abc.ABC):

    @abc.abstractmethod
    def train(self, data):
        """
            Abstract method for training a policy.

            Args:
                data: a dictionary that contains X (observations) and y (actions).
            Returns:
                This method does not return anything. It will just need to update the
                property of a RobotPolicy instance.
        """

    def get_actions(self, observations):
        """
            Abstract method for getting actions. You can do data preprocessing and feed
            forward of your trained model here.
            Args:
                observations: a batch of observations (images or vectors)
            
            Returns:
                A batch of actions with the same batch size as observations.
        """


class Regressor(abc.ABC):

    def train(self, data):
        """A method that train a regressor with given data

           Args:
               data: a dictionary that contains images and the groundtruth location of
                     an agent.
           Returns:
               Nothing
        """
    
    def predict(self, Xs):
        """A method that predicts y's given a batch of X's

           Args:
               Xs: a batch of data (in this project, it is in shape [batch_size, 64, 64, 3])
           Returns:
               The fed-forward results (predicted y's) with a trained model.
        """
