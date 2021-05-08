from base import Regressor
import numpy as np
from sklearn.linear_model import LinearRegression


class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """

    def train(self, data):
        y_t=[]
        for i in range(len(data['info'])):
            c=np.asarray(data['info'][i]['agent_pos'])
            y_t.append(c)

        y=np.array(y_t)
        nsamples, nx, ny, nz = data['obs'].shape
        train_dataset = data['obs'].reshape((nsamples, nx * ny * nz))
        self.model = LinearRegression()
        self.model.fit(train_dataset,y)



    def predict(self, Xs):
        nsamples, nx, ny, nz = Xs.shape
        train_dataset = Xs.reshape((nsamples, nx * ny * nz))
        result=self.model.predict(train_dataset)
        return  result
       



if __name__ == '__main__':
    cc=PositionRegressor()

