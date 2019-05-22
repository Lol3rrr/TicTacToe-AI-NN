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
    tk.Label(self.window, textvariable=self.iterationText).grid(row=self.size, column=int(self.size / 2), padx=5, pady=3)
    self.iterations = 0
    self.iterationText.set(str(self.iterations) + " Iterations")

    self.trainingIterationsText = tk.StringVar()
    tk.Label(self.window, textvariable=self.trainingIterationsText).grid(row=self.size + 1, column=int(self.size / 2), padx=5, pady=3)
    self.trainingIterations = 10000
    self.trainingIterationsText.set(str(self.trainingIterations) + " Training Iterations")

    self.samplesText = tk.StringVar()
    tk.Label(self.window, textvariable=self.samplesText).grid(row=self.size + 2, column=int(self.size / 2), padx=5, pady=3)
    self.samplesText.set("0 Samples")

    # 0 = user
    # 1 = ai
    self.player = 0
    self.starter = 0
    self.playerSteps = []

    self.createNN()

  def createNN(self):
    intputCount = self.size * self.size
    outputCount = self.size * self.size

    self.neural_network = NeuralNetwork(intputCount, intputCount * 6, outputCount, [intputCount * 5, intputCount * 4, intputCount * 3, intputCount * 2])

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

  def guessNN(self):
    before = copy.copy(self.gamePadInfo)

    tmpBoard = []
    for stepY in range(len(before)):
      for stepX in range(len(before[stepY])):
        field = before[stepY][stepX]

        tmpBoard.append(field)

    rawGuesses = self.neural_network.getOutput(tmpBoard)

    return rawGuesses

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

  def getMoveNN(self):
    cords = []

    guessed = False

    rawGuesses = self.guessNN()
    print(rawGuesses)
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

  def addPlayerMove(self, x, y, changePerspectiv):
    before = copy.copy(self.gamePadInfo)
    playerMove = []

    moveBoard = []
    move = []
    for stepY in range(len(before)):
      for stepX in range(len(before[stepY])):
        field = before[stepY][stepX]

        if changePerspectiv:
          if field == 0:
            field = 1
          else:
            if field == 1:
              field = 0

        moveBoard.append(field)
        move.append(0)
        
    move[y * self.size + x] = 1

    playerMove.append(moveBoard)
    playerMove.append(move)

    self.playerSteps.append(playerMove)

    self.samplesText.set(str(len(self.playerSteps)) + " Samples")

  def play(self, x, y):
    if self.player == 1:
      guessedCords = self.getMoveNN()

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
      self.trainNN()
      
      return

    if self.isOver():
      self.restart()
      self.trainNN()

      return