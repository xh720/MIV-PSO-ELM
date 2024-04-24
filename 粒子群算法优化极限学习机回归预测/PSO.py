import numpy as np


class Particle:
    # 进行粒子的初始化

    def __init__(self, dim, max_v, min_v, x_min, x_max, fitness_func):
        # self.dim = len(x_min)  # 获得变量数
        self.dim = dim
        self.max_v = max_v
        self.min_v = min_v
        self.x_min = x_min
        self.x_max = x_max
        '''为了防止不同的变量约束不同，传进来的都是数组'''
        self.fitness_func = fitness_func  # 将 fitness_func 设置为实例属性
        self.pos = np.random.uniform(x_min, x_max, dim)  # 初始化粒子位置
        self._v = np.random.uniform(-max_v, max_v, dim)  # 初始化粒子速度
        self._initialize()  # 初始化并计算适应度
        # self.pos = np.zeros(self.dim)

        # self.pbest = np.zeros(self.dim)
        # self.initPos(x_min, x_max)

        # self._v = np.zeros(self.dim)

        # self.initV(min_v, max_v)  # 初始化速度

    @classmethod
    def _initialize(self):
        self.bestFitness = self.fitness_func(self.pos)  # 使用实例属性调用适应度函数,计算初始适应度
        self.pbest = np.copy(self.pos)  # 设置个体最优位置



    def _updateFit(self):
        fitness = self.fitness(self.pos)
        if fitness < self.bestFitness:
            self.bestFitness = fitness
            self.pbest = np.copy(self.pos)

    # def _updatePos(self):
    #     self.pos = self.pos + self._v
    #     for i in range(self.dim):
    #         self.pos[i] = min(self.pos[i], self.x_max[i])
    #         self.pos[i] = max(self.pos[i], self.x_min[i])
    def _updatePos(self):
        self.pos = np.clip(self.pos + self._v, self.x_min, self.x_max)

    def _updateVel(self, gbest):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = r1 * (self.pbest - self.pos)
        social_velocity = r2 * (gbest - self.pos)
        self._v = self._v + cognitive_velocity + social_velocity
        self._v = np.clip(self._v, -self.max_v, self.max_v)

#     def _updateV(self, w, c1, c2, gbest):
#         '''这里进行的都是数组的运算'''
#         self._v = w * self._v + c1 * np.random.random() * (self.pbest - self.pos) + c2 * np.random.random() * (
#                     gbest - self.pos)
#         for i in range(self.dim):
#             self._v[i] = min(self._v[i], self.max_v[i])
#             self._v[i] = max(self._v[i], self.min_v[i])
#
#     def initPos(self, x_min, x_max):
#         for i in range(self.dim):
#             self.pos[i] = np.random.uniform(x_min[i], x_max[i])
#             self.pbest[i] = self.pos[i]
#
#     def initV(self, min_v, max_v):
#         for i in range(self.dim):
#             self._v[i] = np.random.uniform(min_v[i], max_v[i])
#
#     def getPbest(self):
#         return self.pbest
#
#     def getBestFit(self):
#         return self.bestFitness
#
#     def update(self, w, c1, c2, gbest):
#         self._updateV(w, c1, c2, gbest)
#         self._updatePos()
#         self._updateFit()
#
# class PSO:
#     def __init__(self, pop, generation, x_min, x_max, fitnessFunction, c1=0.1, c2=0.1, w=1):
#         self.c1 = c1
#         self.c2 = c2
#         self.w = w  # 惯性因子
#         # 惯性因子衰减系数
#         self.pop = pop  # 种群大小
#         self.x_min = np.array(x_min)  # 约束
#         self.x_max = np.array(x_max)
#         self.generation = generation
#         self.max_v = (self.x_max - self.x_min) * 0.05
#         self.min_v = -(self.x_max - self.x_min) * 0.05
#         self.fitnessFunction = fitnessFunction
#         # 初始化种群
#         self.particals = [Partical(self.x_min, self.x_max, self.max_v, self.min_v, self.fitnessFunction) for i in
#                           range(self.pop)]
#
#         # 获得全局最佳的信息
#         self.gbest = np.zeros(len(x_min))
#         self.gbestFit = float('Inf')
#
#         self.fitness_list = []  # 每次的最佳适应值
#
#     def init_gbest(self):
#         for part in self.particals:
#             if part.getBestFit() < self.gbestFit:
#                 self.gbestFit = part.getBestFit()
#                 self.gbest = part.getPbest
#
#     def done(self):
#         for i in range(self.generation):
#             for part in self.particals:
#                 part.update(self.w, self.c1, self.c2, self.gbest)
#                 if part.getBestFit() < self.gbestFit:
#                     self.gbestFit = part.getBestFit()
#                     self.gbest = part.getPbest()
#             self.fitness_list.append(self.gbest)
#         return self.fitness_list, self.gbest
#
#     def fitness_function(params, X, y, xt, yt):
#         input_weights, hidden_biases = params
#         elm_model = ELM(input_weights=input_weights, hidden_biases=hidden_biases, activation_func='sigmoid')
#         elm_model.fit(X, y)
#         y_pred = elm_model.predict(xt)
#         mse = np.mean((yt - y_pred) ** 2)
#         return mse
