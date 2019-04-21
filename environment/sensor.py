class Sensor: 

    def __init__(self, reading, good, failure_chance, np_random, functor=False):
        self.good = good
        self.reading = reading
        self.failed = not good and np_random.uniform(0,1) < failure_chance
        self.np_random = np_random
        self.functor = functor
        self.is_noise = bool(np_random.choice(2,1)[0])
        self.systematic_error = self.np_random.uniform(-5,5), self.np_random.uniform(-5,5)


    def sense(self): 
        if self.reading is None:
            return self.failure_code()

        if self.failed:
            return self._sensor_noise() if self.is_noise else self._sensor_systematic() 
        else:
            return self.reading() if self.functor else self.reading

    def _sensor_noise(self):
        return self.np_random.uniform(-1000,1000), self.np_random.uniform(-1000,1000)

    def _sensor_systematic(self):
        reading = (self.reading() if self.functor else self.reading)
        return reading[0] + self.systematic_error[0], reading[1]+ self.systematic_error[1]

    def failure_code(self):
        if not self.failed:
            return 0
        else: 
            return int(self.failed) + int(self.is_noise)