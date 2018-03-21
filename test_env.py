import numpy as np


class EnvAdd(object):
    def __init__(self, A,B):
        #self.observation_space = ObservationSpaceAdd(A,B)
        self.cur_state = np.zeros(4)
        self.A = A
        self.B = B
        self.cur_state[0] = A
        self.cur_state[1] = B
        self.sum = A + B
        self.num_iters = 0

    def reset(self):
        self.cur_state = np.zeros(4)
        self.cur_state[0] = self.A
        self.cur_state[1] = self.B
        self.num_iters = 0
        return self.cur_state

    def step(self, action):
        done = False
        assert(0<= action <=3)
        self.num_iters +=1
        reward = 0
        if action == 0:
            if self.cur_state[3] == self.sum:
                reward = 20
            done = True
        elif action == 1:
            self.cur_state[3] = self.A
            self.cur_state[2] = self.A
        elif action == 2:
            self.cur_state[3] = self.B
            self.cur_state[2] = self.B
        else:
            self.cur_state[2] += 1
            self.cur_state[3] += 1
            reward = 0.5
        if self.num_iters == 10:
            done = True
        return self.cur_state, reward, done,{}


class EnvAdd2(object):
    def __init__(self, A,B):
        #self.observation_space = ObservationSpaceAdd(A,B)
        self.cur_state = np.zeros(4)
        self.A = A
        self.B = B
        self.cur_state[0] = A
        self.cur_state[1] = B
        self.sum = A + B
        self.num_iters = 0

    def reset(self):
        self.cur_state = np.zeros(4)
        self.cur_state[0] = self.A
        self.cur_state[1] = self.B
        self.num_iters = 0
        return self.cur_state

    def step(self, action):
        done = False
        assert(0<= action <=3)
        self.num_iters +=1
        reward = 0
        if action == 0:
            if self.cur_state[3] == self.sum:
                reward = 20
            done = True
        elif action == 1:
            self.cur_state[3] = self.A
            self.cur_state[2] = self.A
            reward = 0.5
        elif action == 2:
            self.cur_state[3] = self.B
            self.cur_state[2] = self.B
            reward = 0.5
        else:
            self.cur_state[2] += 1
            self.cur_state[3] += 1
            reward = 0.2
        if self.num_iters == 10:
            done = True
        return self.cur_state, reward, done,{}
