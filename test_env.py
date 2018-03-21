import numpy as np

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape
        self.state_0 = np.random.randint(0, 50, shape, dtype=np.uint16)
        self.state_1 = np.random.randint(100, 150, shape, dtype=np.uint16)
        self.state_2 = np.random.randint(200, 250, shape, dtype=np.uint16)
        self.state_3 = np.random.randint(300, 350, shape, dtype=np.uint16)
        self.states = [self.state_0, self.state_1, self.state_2, self.state_3]   


class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    Modified 
    """
    def __init__(self, shape=(84, 84, 3)):
        #4 states
        self.rewards = [0.1, -0.2, 0.0, -0.1]
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.action_space = ActionSpace(5)
        self.observation_space = ObservationSpace(shape)
        

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        return self.observation_space.states[self.cur_state]
        

    def step(self, action):
        assert(0 <= action <= 4)
        self.num_iters += 1
        if action < 4:   
            self.cur_state = action
        reward = self.rewards[self.cur_state]
        if self.was_in_second is True:
            reward *= -10
        if self.cur_state == 2:
            self.was_in_second = True
        else:
            self.was_in_second = False
        return self.observation_space.states[self.cur_state], reward, self.num_iters >= 5, {'ale.lives':0}


    def render(self):
        print(self.cur_state)


class ActionSpaceChain(object):
    def __init__(self):
        self.n = 2

    def sample(self):
        return np.random.randint(0, self.n)


class ActionSpaceClique(object):
    def __init__(self,n):
        self.n = n + 1

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpaceChainThermal(object):
    def __init__(self, N):
        self.N = N
        self.shape = [N, 1, 1]

    def get(self, state):
        if state == 0:
            next_state = np.ones(self.N)
        elif state == self.N:
            next_state = np.zeros(self.N)
        else:
            next_state = np.array([0] * state + [1] * (self.N - state))
        return next_state.reshape([-1, 1, 1])

class ObservationSpaceClique(object):
    def __init__(self, N):
        self.N = N
        self.shape = [2*N,1,1]

    def get(self, state):
        next_state = np.zeros(2*self.N)
        next_state[state] = 1
        return next_state.reshape([-1,1,1])


class EnvTestClique(object):
    def __init__(self, N=20):
        self.cur_state = 2
        self.num_iters = 0
        self.action_space = ActionSpaceClique(N)
        self.N = N
        self.observation_space = ObservationSpaceClique(N)

    def reset(self):
        self.cur_state = 1
        self.num_iters = 0
        return self.observation_space.get(self.cur_state)

    def step(self, action):
        assert(0<= action <= (self.N + 1))
        self.num_iters += 1
        if self.cur_state <= (self.N - 1) and action <= (self.N - 1):
            reward = 0.01
            self.cur_state = action
        elif self.cur_state >= self.N and action <= (self.N - 1):
            reward = 1
            self.cur_state = action + self.N
        elif self.cur_state == (self.N -1) and action == self.N:
            reward = -0.1
            self.cur_state = action
        elif self.cur_state == self.N and action == self.N:
            reward = -0.1
            self.cur_state = action-1
        else:
            reward = 0.01
        return self.observation_space.get(self.cur_state), reward, self.num_iters > self.N + 9, {}

    def render(self):
        print(self.cur_state)


class EnvTestChain(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    Modified 
    """
    def __init__(self, N=50):
        self.cur_state = 2
        self.num_iters = 0
        self.action_space = ActionSpaceChain()
        self.N = N
        self.observation_space = ObservationSpaceChainThermal(N)

    def reset(self):
        self.cur_state = 2
        self.num_iters = 0
        return self.observation_space.get(self.cur_state)
        

    def step(self, action):
        """
            0 go small, 1 go large
        """
        assert(action == 0 or action == 1)
        self.num_iters += 1
        if self.cur_state <= 1 and action == 0:
            reward = 0.001
        elif self.cur_state == self.N and action == 1:
            reward = 1
        else:
            reward = 0
        if action == 0:
            self.cur_state = self.cur_state - 1
        else:
            self.cur_state = self.cur_state + 1
        self.cur_state = max(self.cur_state, 1)
        self.cur_state = min(self.cur_state, self.N)
        return self.observation_space.get(self.cur_state), reward, self.num_iters >= self.N + 9, {}

    def render(self):
        print(self.cur_state)


#class ActionSpaceAdd(object):
#    def __init__(self):
#        self.n = 6
#
#class ObservationSpaceAdd(object):
#    def __init__(self, A,B):
#        self.states = np.zeros(4)
#        self.states[0] = A
#        self.states[1] = B

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
            #reward = 0.5
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