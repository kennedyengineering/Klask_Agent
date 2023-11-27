# Simple human controlled agent to interact with the simulation

class KeyboardController():
    __position_x = 0
    __position_y = 0

    def __init__(self, force):
        self.force = force

    def getAction(self):
        return (self.__position_x * self.force, self.__position_y * self.force)

    def keyUp_pressed(self):
        self.__position_y += 1

    def keyUp_released(self):
        self.__position_y -= 1

    def keyDown_pressed(self):
        self.__position_y -= 1

    def keyDown_released(self):
        self.__position_y += 1

    def keyLeft_pressed(self):
        self.__position_x -= 1

    def keyLeft_released(self):
        self.__position_x += 1

    def keyRight_pressed(self):
        self.__position_x += 1

    def keyRight_released(self):
        self.__position_x -= 1
