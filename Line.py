class Line:
    ''' mnozi se sa 1.0 svuda da bi pretvorili u float'''

    def __init__(self, x1, y1, x2, y2):
        self.x1 = 1.0 * x1
        self.y1 = 1.0 * y1
        self.x2 = 1.0 * x2
        self.y2 = 1.0 * y2

    def getN(self):
        return 1.0 * self.getK() * (-self.x1) + self.y1

    def getK(self):
        return 1.0 * (self.y2 - self.y1) / (self.x2 - self.x1)
