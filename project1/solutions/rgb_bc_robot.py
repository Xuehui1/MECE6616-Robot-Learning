from base import RobotPolicy
from sklearn.decomposition import PCA
from sklearn.linear_model import  LogisticRegression
class RGBBCRobot(RobotPolicy):

    """ Implement solution for Part3 below """

    def train(self, data):
        self.pca = PCA(n_components=10, svd_solver= 'full')
        self.clf= LogisticRegression()

        nsamples, nx, ny, nz = data['obs'].shape
        x1 = data['obs'].reshape((nsamples, nx * ny * nz))
        
        x1=self.pca.fit_transform(x1)
        y1=data['actions']

        self.clf=self.clf.fit(x1,y1)


    def get_actions(self, observations):
    	observations=self.pca.transform(observations)
    	return self.clf.predict(observations)


if __name__ == '__main__':
   cc=RGBBCRobot()


