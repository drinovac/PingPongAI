import pygame as pg
import random
import numpy as np
import math

from NeuralNetwork import NeuralNetwork

SCREEN_WIDTH = 1150
SCREEN_HEIGHT = 700

pc = 0.6
pm = 0.1

from pygame.locals import (
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT
)


class Game():
    width = SCREEN_WIDTH
    height = SCREEN_HEIGHT
    gameDisplay = pg.display.set_mode((width, height))
    population = 100
    generation = 1
    boards = []
    deadBoards = []
    killed = 0

    def __init__(self):
        pg.init()
        self.clock = pg.time.Clock()
        self.board = Board()
        self.gameLoop()

    def gameLoop(self):
        running = True
        font = pg.font.SysFont(None,25)
        for i in range(self.population):
            self.boards.append(Board())

        while running:
            

            msg = 'Generation: ' + str(self.generation)
            screen_text = font.render(msg,True,(0,0,0))
            self.gameDisplay.blit(screen_text,[10,10])

            for board in self.boards:
                board.predict()
                board.updateBall()

                for event in pg.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            running = False
                        elif event.type == pg.QUIT:
                            running = False
                        elif event.key == K_RIGHT:
                            board.ball_y = SCREEN_HEIGHT

                if(board.hitSides()):
                    board.ball_vel_x *= -1
                if(board.hitFloor()):
                    board.ball_vel_y *= -1
                if(board.hitBoard()):
                    board.ball_vel_y *= -1
                    board.score += 1
                if(board.dead()):
                    board.fitness += board.distFromDeadToBoard()
                    self.deadBoards.append(board)
                    self.boards.remove(board)
                    if(len(self.boards) == 0):
                        GA(self).nextGen()
                        self.generation += 1
                else:
                    board.showBall(board.ball_x, board.ball_y)
                    board.showBoard(board.board_x, board.board_y)
                
                        

            pg.display.update()
            self.gameDisplay.fill((210,210,210))
            self.clock.tick(1000)
        pg.quit()
        quit()

class Board:

    def __init__(self):
        self.length = 160
        self.height = 26
        self.board_x = (SCREEN_WIDTH - self.length) / 2
        self.board_y = SCREEN_HEIGHT - self.height * 2
        self.radius = 16
        self.ball_x = random.randrange(17, SCREEN_WIDTH - 17)
        self.ball_y = SCREEN_HEIGHT / 2
        self.ball_vel_x = random.random() + 0.5
        self.ball_vel_y = random.random() - 1.5
        self.bar_vel = 0
        self.score = 0
        self.fitness = 0
        self.distance = 0
        self.brain = NeuralNetwork(7,4,2)


    def showBoard(self, x, y):
        pg.draw.rect(Game.gameDisplay, (0,0,0), [x,y,self.length,self.height], border_radius=13)

    def showBall(self, x, y):
        pg.draw.circle(Game.gameDisplay, (0,0,0), (int(x), int(y)), self.radius)

    def updateBall(self):
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

    def hitSides(self):
        if(self.ball_x < self.radius or self.ball_x + self.radius > SCREEN_WIDTH):
            return True
            
    def hitFloor(self):
        if(self.ball_y <= 0):
            return True
            
    def hitBoard(self):
        if(abs((self.ball_y + self.radius) - (self.board_y + self.height / 2)) < 0.5 and self.board_x + self.length > self.ball_x + 5  and self.board_x < self.ball_x - 5):
            return True

    def dead(self):
        if(self.ball_y + self.radius > SCREEN_HEIGHT):
            return True

    def moveLeft(self):
        self.board_x -= 10
        self.distance += 1
        if(self.board_x < 0):
            self.board_x = 0
            self.distance -= 1
        
    def moveRight(self):
        self.board_x += 10
        self.distance += 1
        if(self.board_x + self.length > SCREEN_WIDTH):
            self.board_x = SCREEN_WIDTH - self.length
            self.distance -= 1

    def CalcDistance(self):
        return math.sqrt((self.board_x - self.ball_x)**2 + (self.board_y - self.ball_y)**2)

    def predict(self):

        boardLeft = self.board_x / SCREEN_WIDTH
        boardRight = 1 - (SCREEN_WIDTH + self.board_x - self.length) / SCREEN_WIDTH
        ballLeft = self.ball_x / SCREEN_WIDTH
        ballFloor = self.ball_y / SCREEN_HEIGHT
        ballVel_x = self.ball_vel_x / 10
        ballVel_y = self.ball_vel_y / 10

        distIQuadr = self.board_x + self.length / 2 - self.ball_x

        disty = self.board_y - self.ball_y
        
        distIQuadr = math.atan2(disty, distIQuadr) / math.atan2(1,0) * 0.4
        #distIQuadr = math.atan2(disty, distIQuadr) / 10
        distCenter = abs((self.board_x + self.length / 2) - (SCREEN_WIDTH / 2))

        dist = self.CalcDistance()

        ins = [boardLeft, boardRight, ballLeft, ballFloor, distIQuadr, ballVel_x, ballVel_y]
        inputs = np.array(ins)
        inputs = np.reshape(inputs,(7,1))
        output = self.brain.feedforward(inputs)
        

        if(output[0] > output[1]):
            self.moveLeft()
        else:
            self.moveRight()
    
    def distFromDeadToBoard(self):
        dist = abs(self.board_x - self.ball_x)

        if dist < SCREEN_WIDTH / 10:
            return 50
        elif dist < SCREEN_WIDTH / 5:
            return 40
        elif dist < 3 * SCREEN_WIDTH / 10:
            return 30
        elif dist < 2 * SCREEN_WIDTH / 5:
            return 20
        elif dist < SCREEN_WIDTH / 2 :
            return 10
        else:
            return 0

    def calcFitness(self):
        return self.score**2 + self.distance**4 + self.fitness

class GA(Game):

    def __init__(self, game):
        self.game = game

    def nextGen(self):
        
        boards = self.deadBoards
        scores = []
        for board in boards:
            scores.append(board.calcFitness())
        
        self.game.boards = self.crossOver(boards)
        self.boards = self.game.boards
        
        for board in self.game.boards:
            board.fitness = 0

        self.game.deadBoards = []
        self.deadBoards = []

        return

    def getBest10(self, boards):
        
        boardsCopy = boards.copy()
        best10 = []
        
        for i in range(10):
            bestFitness = 0
            bestIndex = 0
            bestBoard = Board()
            i = 0
            for board in boardsCopy:
                
                if(board.calcFitness() > bestFitness):
                    bestFitness = board.calcFitness()
                    bestIndex = i
                    bestBoard = board
                
                i += 1

            bestBoard.ball_x = random.randrange(15, SCREEN_WIDTH - 15)
            bestBoard.ball_y = SCREEN_HEIGHT / 2
            bestBoard.board_x = (SCREEN_WIDTH - bestBoard.length) / 2
            bestBoard.board_y = SCREEN_HEIGHT - bestBoard.height * 2

            if(bestBoard.ball_vel_y > 0) : 
                bestBoard.ball_vel_y *= -1

            bestBoard.score = 0
            bestBoard.distance = 0

            best10.append(bestBoard)
            boardsCopy.pop(bestIndex)

        return best10


    def tournament5(self):

        ktournament = []
        bestFitness = 0
        bestBoard = Board()
        for j in range(5):
            randBoard = self.deadBoards[random.randint(0, self.population - 1)]
            
            ktournament.append(randBoard)
            if(randBoard.calcFitness() > bestFitness):
                bestFitness = randBoard.calcFitness()
                bestBoard = randBoard

        bestBoard.ball_x = random.randrange(17, SCREEN_WIDTH - 17)
        bestBoard.ball_y = SCREEN_HEIGHT / 2
        bestBoard.board_x = (SCREEN_WIDTH - bestBoard.length) / 2
        bestBoard.board_y = SCREEN_HEIGHT - bestBoard.height * 2

        if(bestBoard.ball_vel_y > 0) : 
            bestBoard.ball_vel_y *= -1

        bestBoard.score = 0
        bestBoard.distance = 0

        return bestBoard

            
        return newGen
    
    def crossOver(self, boards):
    
        newGen = []

        for board in self.getBest10(boards):
            newGen.append(board)
        

        for i in range(self.game.population - 10):
            firstParent = self.tournament5()
            secondParent = self.tournament5()

            child = Board()

            board1_in_hidden = firstParent.brain.in_hidden_weights
            board2_in_hidden = secondParent.brain.in_hidden_weights
            cross = np.concatenate((board1_in_hidden[:1], board2_in_hidden[1:2], board1_in_hidden[2:3], board2_in_hidden[3:4]))
            child.brain.in_hidden_weights = cross

            board1_hidden_output = firstParent.brain.hidden_output_weights
            board2_hidden_output = secondParent.brain.hidden_output_weights
            cross = np.concatenate((board1_hidden_output[0:1], board2_hidden_output[1:2]))
            child.brain.hidden_output_weights = cross
        
                        
            child.brain.in_hidden_biases = firstParent.brain.in_hidden_biases

            child.brain.hidden_out_biases = secondParent.brain.hidden_out_biases

            self.mutate(child)

            newGen.append(child)

        return newGen

        
        """for i in range(half):
            if(random.random() <= pc):
                
                board1_in_hidden = firstParent.brain.in_hidden_weights
                board2_in_hidden = secondParent.brain.in_hidden_weights
                cross1 = np.concatenate((board1_in_hidden[:2], board2_in_hidden[2:5]))
                cross2 = np.concatenate((board2_in_hidden[:2], board1_in_hidden[2:5]))
                firstParent.brain.in_hidden_weights = cross1
                self.game.boards[half + 1].brain.in_hidden_weights = cross2

                board1_hidden_output = firstParent.brain.hidden_output_weights
                board2_hidden_output = secondParent.brain.hidden_output_weights
                cross1 = np.concatenate((board1_hidden_output[0:1], board2_hidden_output[1:2]))
                cross2 = np.concatenate((board2_hidden_output[0:1], board1_hidden_output[1:2]))
                firstParent.brain.hidden_output_weights = cross1
                self.game.boards[half + 1].brain.hidden_output_weights = cross2

                board1_in_hidden_biases = firstParent.brain.in_hidden_biases
                board2_in_hidden_biases = secondParent.brain.in_hidden_biases
                cross1 = board1_in_hidden_biases
                cross2 = board2_in_hidden_biases
                firstParent.brain.in_hidden_biases = cross2
                self.game.boards[half + 1].brain.in_hidden_biases = cross1

                board1_hidden_out_biases = firstParent.brain.hidden_out_biases
                board2_hidden_out_biases = secondParent.brain.hidden_out_biases
                cross1 = board1_hidden_out_biases
                cross2 = board2_hidden_out_biases
                firstParent.brain.hidden_out_biases = cross2
                self.game.boards[half + 1].brain.hidden_out_biases = cross1 """


    def mutate(self, child):

        for i in range(4):
            for j in range(7):
                if(random.random() < pm):
                    child.brain.in_hidden_weights[i][j] = random.random() * 2 - 1
            
        for i in range(2):
            for j in range(4):
                if(random.random() < pm):
                    child.brain.hidden_output_weights[i][j] = random.random() * 2 - 1

        for i in range(4):
            for j in range(1):
                if(random.random() < pm):
                    child.brain.in_hidden_biases[i][j] = random.random() * 2 - 1
        
        for i in range(2):
            for j in range(1):
                if(random.random() < pm):
                    child.brain.hidden_out_biases[i][j] = random.random() * 2 - 1



        """for board in self.game.boards:
            board_in_hidden = board.brain.in_hidden_weights
            for i in range(5):
                for j in range(9):
                    if(random.random() < pm):
                        board_in_hidden[i][j] = random.random()

            board_hidden_output_weights = board.brain.hidden_output_weights
            for i in range(2):
                for j in range(5):
                    if(random.random() < pm):
                        board_hidden_output_weights[i][j] = random.random()

            board_in_hidden_biases = board.brain.in_hidden_biases
            for i in range(5):
                for j in range(1):
                    if(random.random() < pm):
                        board_in_hidden_biases[i][j] = random.random()

            board_hidden_out_biases = board.brain.hidden_out_biases
            for i in range(2):
                for j in range(1):
                    if(random.random() < pm):
                        board_hidden_out_biases[i][j] = random.random()"""        

if __name__ == '__main__':
    game = Game()
