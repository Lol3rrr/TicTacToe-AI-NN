# Custom Class
from nn import NeuralNetwork

#
import copy
import numpy as np
import tkinter as tk
from functools import partial

class Game:
  def __init__(self, gameSize):
    self.size = gameSize
    self.plays = 0

    self.window = tk.Tk()

    self.setup()

  def setup(self):
    self.gamePadInfo = []
    self.gamePadButtonText = []
    for yIteration in range(self.size):
      tmpInfos = []
      tmpTexts = []

      for xIteration in range(self.size):
        btn_text = tk.StringVar()
        button = tk.Button(self.window, textvariable=btn_text, command = partial(self.play, xIteration, yIteration), height=3, width=6).grid(row=xIteration, column=yIteration, padx=5, pady=5)

        btn_text.set("-")

        tmpInfos.append(-1)
        tmpTexts.append(btn_text)
      
      self.gamePadInfo.append(tmpInfos)
      self.gamePadButtonText.append(tmpTexts)

    self.iterationText = tk.StringVar()
    tk.Label(self.window, textvariable=self.iterationText).grid(row=0, column=self.size, padx=5, pady=3)
    self.iterations = 0
    self.iterationText.set(str(self.iterations) + " Iterations")

    self.trainingIterationsText = tk.StringVar()
    tk.Label(self.window, textvariable=self.trainingIterationsText).grid(row=1, column=self.size, padx=5, pady=3)
    self.trainingIterations = 10000
    self.trainingIterationsText.set(str(self.trainingIterations) + " Training Iterations")

    self.samplesText = tk.StringVar()
    tk.Label(self.window, textvariable=self.samplesText).grid(row=2, column=self.size, padx=5, pady=3)
    self.samplesText.set("0 Samples")

    self.trainingRoundsInput = tk.StringVar()
    trainingInput = tk.Entry(self.window, textvariable=self.trainingRoundsInput).grid(row=0, column=self.size + 1, padx=5, pady=5)
    trainMatch = tk.Button(self.window, text="Train NNs", command = partial(self.trainMatchNN)).grid(row=1, column=self.size + 1, padx=5, pady=5)

    tk.Button(self.window, text="Save", command = partial(self.saveNN)).grid(row=0, column=self.size + 2, padx=5, pady=5)
    tk.Button(self.window, text="Load", command = partial(self.loadNN)).grid(row=1, column=self.size + 2, padx=5, pady=5)

    # 0 = user
    # 1 = ai
    self.player = 0
    self.starter = 0
    self.playerSteps = []

    self.createNN()

  def getBoard(self):
    before = copy.copy(self.gamePadInfo)

    tmpBoard = []
    for stepY in range(len(before)):
      for stepX in range(len(before[stepY])):
        field = before[stepY][stepX]

        tmpBoard.append(field)

    return tmpBoard

  def createNN(self):
    intputCount = self.size * self.size
    outputCount = self.size * self.size

    self.neural_network = NeuralNetwork(intputCount, intputCount * 6, outputCount, [intputCount * 5, intputCount * 4, intputCount * 3, intputCount * 2])

  def saveNN(self):
    print("Saving Neural Network...")

    self.neural_network.save()

  def loadNN(self):
    print("Loading Neural Network...")

    worked = self.neural_network.load()

    if worked == False:
      print("Could not load Neural Network")

  def trainNN(self):
    inputs = []
    outputs = []

    for iteration in range(len(self.playerSteps)):
      step = self.playerSteps[iteration]

      inputs.append(step[0])
      outputs.append(step[1])

    print("Training")

    self.neural_network.train(np.array(inputs), np.array(outputs), self.trainingIterations)

    self.iterations += self.trainingIterations
    self.iterationText.set(str(self.iterations) + " Iterations")

  def guessNN(self, changePerspectiv):
    tmpBoard = self.getBoard()

    if changePerspectiv:
      for index in range(len(tmpBoard)):
        if tmpBoard[index] == 0:
          tmpBoard[index] = 1
        else:
          if tmpBoard[index] == 1:
            tmpBoard[index] = 0

    rawGuesses = self.neural_network.getOutput(tmpBoard)

    return rawGuesses

  def getMoveNN(self, changePerspectiv):
    cords = []

    guessed = False

    rawGuesses = self.guessNN(changePerspectiv)
    guesses = rawGuesses.tolist()
    sortedGuesses = copy.copy(guesses)
    sortedGuesses.sort()
        
    index = len(sortedGuesses) - 1
    while True:
      value = sortedGuesses[index]
      valueIndex = guesses.index(value)

      before = copy.copy(self.gamePadInfo)
      for stepY in range(len(before)):
        for stepX in range(len(before[stepY])):
          if (stepY * self.size + stepX) == valueIndex:
            x = stepX
            y = stepY

      if self.gamePadInfo[y][x] == -1:
        cords.append(x)
        cords.append(y)
        return cords

      index -= 1

    return [-1, -1]

  def trainMatchNN(self):
    defaultTrainingRounds = 10

    trainingRounds = int(self.trainingRoundsInput.get())

    if trainingRounds < 3:
      trainingRounds = defaultTrainingRounds

      self.trainingRoundsInput.set("10")

    for currentRound in range(trainingRounds):
      tmpPlayer0Moves = []
      tmpPlayer1Moves = []
      self.player = 0
      while True:
        if self.player == 1:
          guessedCords = self.getMoveNN(False)

          x = guessedCords[0]
          y = guessedCords[1]

          tmpPlayer0Moves.append(self.createPlayerMove(x, y, False))

          self.gamePadButtonText[y][x].set("O")
          self.gamePadInfo[y][x] = self.player

          self.player = 0

        else:
          guessedCords = self.getMoveNN(True)

          x = guessedCords[0]
          y = guessedCords[1]

          tmpPlayer1Moves.append(self.createPlayerMove(x, y, True))

          self.gamePadButtonText[y][x].set("X")
          self.gamePadInfo[y][x] = self.player

          self.player = 1

        winner = self.checkWin()
        if winner != -1:
          print("Winner " + str(winner))
          self.restart()

          if winner == 0:
            for iteration in range(len(tmpPlayer0Moves)):
              self.playerSteps.append(tmpPlayer0Moves[iteration])
          else:
            for iteration in range(len(tmpPlayer1Moves)):
              self.playerSteps.append(tmpPlayer1Moves[iteration])

          self.trainNN()
          
          break

        if self.isOver():
          print("Tie")
          self.restart()

          for iteration in range(len(tmpPlayer0Moves)):
            self.playerSteps.append(tmpPlayer0Moves[iteration])
          for iteration in range(len(tmpPlayer1Moves)):
            self.playerSteps.append(tmpPlayer1Moves[iteration])

          self.trainNN()

          break
      
      self.iterationText.set(str(self.iterations) + " Iterations")
      self.samplesText.set(str(len(self.playerSteps)) + " Samples")
      self.window.update()

      print("Done with Iteration: " + str(currentRound))

      if currentRound % 10 == 0:
        self.saveNN()

    self.saveNN()

  def restart(self):
    self.plays = self.plays + 1

    for y in range(len(self.gamePadButtonText)):
      for x in range(len(self.gamePadButtonText[y])):
        self.gamePadButtonText[y][x].set("-")
        self.gamePadInfo[y][x] = -1

    if self.starter == 0:
      self.starter = 1
    else:
      self.starter = 0
    self.player = self.starter

  def run(self):
    self.window.mainloop()

  def checkWin(self):
    # Checking for a horizontal line
    for y in range(self.size):
      player = self.gamePadInfo[y][0]
      for x in range(self.size):
        if self.gamePadInfo[y][x] != player:
          break
        
        if x == self.size - 1:
          return player

    # Checking for a vertical line
    for x in range(self.size):
      player = self.gamePadInfo[0][x]
      for y in range(self.size):
        if self.gamePadInfo[y][x] != player:
          break
        
        if y == self.size - 1:
          return player

    # Checking for a diagonal line from top left
    for iteration in range(self.size):
      player = self.gamePadInfo[0][0]
      if self.gamePadInfo[iteration][iteration] != player:
        break
        
      if iteration == self.size - 1:
        return player

    # Checking for a diagonal line from top right
    for iteration in range(self.size):
      lastIndex = self.size - 1
      player = self.gamePadInfo[0][lastIndex]
      if self.gamePadInfo[iteration][lastIndex - iteration] != player:
        break
        
      if iteration == self.size - 1:
        return player

    return -1

  def isOver(self):
    for y in range(self.size):
      for x in range(self.size):
        if self.gamePadInfo[y][x] == -1:
          return False
    
    return True

  def createPlayerMove(self, x, y, changePerspectiv):
    playerMove = []

    move = []
    moveBoard = self.getBoard()
    for index in range(len(moveBoard)):
      if changePerspectiv:
        if moveBoard[index] == 0:
          moveBoard[index] = 1
        else:
          if moveBoard[index] == 1:
            moveBoard[index] = 0
      
      move.append(0)

    move[y * self.size + x] = 1

    playerMove.append(moveBoard)
    playerMove.append(move)

    return playerMove

  def addPlayerMove(self, x, y, changePerspectiv):
    playerMove = self.createPlayerMove(x, y, changePerspectiv)

    self.playerSteps.append(playerMove)

    self.samplesText.set(str(len(self.playerSteps)) + " Samples")

  def play(self, x, y):
    if self.player == 1:
      guessedCords = self.getMoveNN(False)

      x = guessedCords[0]
      y = guessedCords[1]

      self.gamePadButtonText[y][x].set("O")
      self.gamePadInfo[y][x] = self.player

      self.player = 0

    else:
      if self.gamePadInfo[y][x] != -1:
        return

      self.addPlayerMove(x, y, True)

      self.gamePadButtonText[y][x].set("X")
      self.gamePadInfo[y][x] = self.player

      self.player = 1

    winner = self.checkWin()
    if winner != -1:
      print("Winner " + str(winner))
      self.restart()
      
      return

    if self.isOver():
      self.restart()

      return