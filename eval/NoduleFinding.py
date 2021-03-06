class NoduleFinding(object):
    '''
    Represents a nodule
    TODO: 改名
    '''

    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
                 CADprobability=None, noduleType=None, diameter=None, state=None, seriesInstanceUID=None):
        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        self.diameter_mm = diameter
        self.state = state
        self.candidateID = None
        self.seriesuid = seriesInstanceUID

    def included(self):
        return self.state == 'Included'

    def excluded(self):
        return self.state == 'Excluded'