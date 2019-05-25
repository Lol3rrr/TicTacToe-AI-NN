# Custom Class
from qLearning import QTable

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
    # 1 = user
    # 2 = ai
    self.player = 1
    self.starter = 1
    self.empty = 0
    self.gamePadInfo = []
    self.gamePadButtonText = []

    for yIteration in range(self.size):
      tmpInfos = []
      tmpTexts = []

      for xIteration in range(self.size):
        btn_text = tk.StringVar()
        button = tk.Button(self.window, textvariable=btn_text, command = partial(self.play, xIteration, yIteration), height=3, width=6).grid(row=xIteration, column=yIteration, padx=5, pady=5)

        btn_text.set("-")

        tmpInfos.append(self.empty)
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
    trainMatch = tk.Button(self.window, text="Train NNs", command = partial(self.trainMatchQ)).grid(row=1, column=self.size + 1, padx=5, pady=5)

    tk.Button(self.window, text="Save", command = partial(self.saveQ)).grid(row=0, column=self.size + 2, padx=5, pady=5)
    tk.Button(self.window, text="Load", command = partial(self.loadQ)).grid(row=1, column=self.size + 2, padx=5, pady=5)

    self.createQ()

  def getBoard(self):
    before = copy.copy(self.gamePadInfo)

    tmpBoard = []
    for stepY in range(len(before)):
      for stepX in range(len(before[stepY])):
        field = before[stepY][stepX]

        tmpBoard.append(field)

    return tmpBoard

  def createQ(self):
    self.qTable = QTable(self.size, 0.05, 0.5)

  def saveQ(self):
    print("Saving QTable...")

    self.qTable.save()

  def loadQ(self):
    print("Loading QTable...")

    worked = self.qTable.load()

    if worked == False:
      print("Could not load QTable")

  def evaluteQ(self, result, steps):
    moveCount = len(steps)
    reward = QTable.getReward(result, moveCount)
    for iteration in range(moveCount):
      step = steps[iteration]

      board = step[0]
      outputIndex = step[1][0]

      self.qTable.evaluate(board, outputIndex, reward)

  def getMoveQ(self, changePerspectiv, training):
    board = self.getBoard()

    if changePerspectiv == True:
      for iteration in range(len(board)):
        value = board[iteration]

        if value == 1:
          board[iteration] = 2
        else:
          if value == 2:
            board[iteration] = 1

    valueIndex = self.qTable.getOutput(board, training)

    while True:
      before = copy.copy(self.gamePadInfo)
      for stepY in range(len(before)):
        for stepX in range(len(before[stepY])):
          if (stepY * self.size + stepX) == valueIndex:
            x = stepX
            y = stepY

      if self.gamePadInfo[y][x] == self.empty:
        return [x, y]

      # Tell it that, the choosen move cant be made
      self.qTable.evaluate(board, valueIndex, -100)
      valueIndex = self.qTable.getOutput(board, training)

    return [-1, -1]

  def trainMatchQ(self):
    defaultTrainingRounds = 10

    trainingRounds = int(self.trainingRoundsInput.get())
    print(str(trainingRounds))

    if trainingRounds < -1:
      trainingRounds = defaultTrainingRounds

      self.trainingRoundsInput.set("10")

    roundCount = trainingRounds
    if trainingRounds == -1:
      roundCount = 1

    currentRound = 0
    iteration = 0
    while currentRound < roundCount:
      steps = []
      while True:
        # AI
        if self.player == 2:
          guessedCords = self.getMoveQ(False, True)

          x = guessedCords[0]
          y = guessedCords[1]

          preBoard = self.getBoard()
          outputIndex = y * self.size + x
          steps.append([preBoard, [outputIndex]])

          self.gamePadButtonText[y][x].set("O")
          self.gamePadInfo[y][x] = self.player

          self.player = 1

        # Usually Human
        else:
          guessedCords = self.getMoveQ(True, True)

          x = guessedCords[0]
          y = guessedCords[1]

          self.gamePadButtonText[y][x].set("X")
          self.gamePadInfo[y][x] = self.player

          self.player = 2

        winner = self.checkWin()
        if winner != -1:
          self.restart()

          if winner == 1:
            self.evaluteQ(1, steps)
          else:
            self.evaluteQ(-1, steps)
          
          break

        if self.isOver():
          self.restart()

          self.evaluteQ(0, steps)

          break
      
      self.iterations += 1
      self.iterationText.set(str(self.iterations) + " Iterations")
      self.window.update()

      if iteration % 2500 == 0:
        self.saveQ()

      currentRound += 1
      iteration += 1

      if trainingRounds == -1:
        currentRound -= 1


    self.saveQ()

  def restart(self):
    self.plays = self.plays + 1

    for y in range(len(self.gamePadButtonText)):
      for x in range(len(self.gamePadButtonText[y])):
        self.gamePadButtonText[y][x].set("-")
        self.gamePadInfo[y][x] = self.empty

    if self.starter == 1:
      self.starter = 2
    else:
      self.starter = 1
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
          if player != self.empty:
            return player

    # Checking for a vertical line
    for x in range(self.size):
      player = self.gamePadInfo[0][x]
      for y in range(self.size):
        if self.gamePadInfo[y][x] != player:
          break
        
        if y == self.size - 1:
          if player != self.empty:
            return player

    # Checking for a diagonal line from top left
    for iteration in range(self.size):
      player = self.gamePadInfo[0][0]
      if self.gamePadInfo[iteration][iteration] != player:
        break
        
      if iteration == self.size - 1:
        if player != self.empty:
            return player

    # Checking for a diagonal line from top right
    for iteration in range(self.size):
      lastIndex = self.size - 1
      player = self.gamePadInfo[0][lastIndex]
      if self.gamePadInfo[iteration][lastIndex - iteration] != player:
        break
        
      if iteration == self.size - 1:
        if player != self.empty:
            return player

    return -1

  def isOver(self):
    for y in range(self.size):
      for x in range(self.size):
        if self.gamePadInfo[y][x] == self.empty:
          return False
    
    return True

  def play(self, x, y):
    if self.player == 2:
      guessedCords = self.getMoveQ(False, False)

      x = guessedCords[0]
      y = guessedCords[1]

      self.gamePadButtonText[y][x].set("O")
      self.gamePadInfo[y][x] = self.player

      self.player = 1

    else:
      if self.gamePadInfo[y][x] != self.empty:
        return

      self.gamePadButtonText[y][x].set("X")
      self.gamePadInfo[y][x] = self.player

      self.player = 2

    winner = self.checkWin()
    if winner != -1:
      print("Winner " + str(winner))
      self.restart()
      
      return

    if self.isOver():
      print("Over")
      self.restart()

      return