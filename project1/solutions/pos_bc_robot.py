from base import RobotPolicy
from sklearn.svm import SVC
class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """

    def train(self, data):
        self.model = SVC(C=1,kernel='poly', coef0=0.5, degree=1)
        self.model.fit(data['obs'], data['actions'])
        

    def get_actions(self, observations):
        result=self.model.predict(observations)
        return result


if __name__ == '__main__':
    cc=POSBCRobot()
