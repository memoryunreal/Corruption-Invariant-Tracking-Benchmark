# input pertubations
import numpy as np
class Robustness(object):
    def __init__(self, region) -> None:
        super().__init__()
        self.x_lt = region[0]
        self.y_lt = region[1]
        self.w = region[2]
        self.h = region[3]
      
    def functions(self, type):
        if type == 1:
            self.sre_1()
        elif type == 2:    
            self.sre_2()
        
        elif type == 3:    
            self.sre_3()
        elif type == 4:    
            self.sre_4()
        elif type == 5:    
            self.sre_5()
        elif type == 6:    
            self.sre_6()
        elif type == 7:    
            self.sre_7()
        elif type == 8:    
            self.sre_8()
        elif type == 0:
            self.sre_0()
        elif type == 80:
            self.sre_80()
        elif type == 90:
            self.sre_90()
        elif type == 110:
            self.sre_110()
        elif type == 120:
            self.sre_120()
    # normal
    def sre_0(self):
        print('no change')

    #center left shift 10%
    def sre_1(self):
        self.x_lt = np.max((self.x_lt - 0.1* self.w, 0))


    #center right shift 10%
    def sre_2(self):
        self.x_lt = np.max((self.x_lt + 0.1* self.w, 0))
        print('shirft')
    #center top shift 10%
    def sre_3(self):
        self.y_lt = np.max((self.y_lt - 0.1* self.h, 0))

    #center bottom shift 10%
    def sre_4(self):
        self.y_lt = np.max((self.y_lt + 0.1* self.h, 0))
    
    #corner left-top shift 10%
    def sre_5(self):
        self.x_lt = np.max((self.x_lt - 0.1 * self.w, 0))
        self.y_lt = np.max((self.y_lt - 0.1 * self.h, 0))
    
    #corner left-bottom shift 10%
    def sre_6(self):
        self.x_lt = np.max((self.x_lt - 0.1 * self.w, 0))
        self.y_lt = np.max((self.y_lt + 0.1 * self.h, 0))
    
    #corner right-top shift 10%
    def sre_7(self):
        self.x_lt = np.max((self.x_lt + 0.1 * self.w, 0))
        self.y_lt = np.max((self.y_lt - 0.1 * self.h, 0))
    
    #corner right-bottom shift 10%
    def sre_8(self):
        self.x_lt = np.max((self.x_lt + 0.1 * self.w, 0))
        self.y_lt = np.max((self.y_lt + 0.1 * self.h, 0))
    
    # bbox scale 80%
    def sre_80(self):
        self.w = 0.8 * self.w
        self.h = 0.8 * self.h
    
    # bbox scale 90%
    def sre_90(self):
        self.w = 0.9 * self.w
        self.h = 0.9 * self.h
    
    # bbox scale 110%
    def sre_110(self):
        self.w = 1.1 * self.w
        self.h = 1.1 * self.h
    
    # bbox scale 120%
    def sre_120(self):
        self.w = 1.2 * self.w
        self.h = 1.2 * self.h
    
    def case_fun_other(self, msg):
        print(msg)

    @property
    def region(self):
        region = [self.x_lt, self.y_lt, self.w, self.h]
        return region


