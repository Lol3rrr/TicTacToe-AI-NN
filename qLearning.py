import random
import pickle

class QTable():
  def __init__(self, boardSize, learningRate, discountRate):
    self.tableFile = "qTable.dat"
    self.statusFile = "qStatus.dat"
    
    self.table = []
    self.states = []
    self.boardSize = boardSize

    self.maxReward = 100
    self.learningRate = learningRate
    self.discountRate = discountRate

    for iteration in range(3 ** (self.boardSize ** 2)):
      actions = []
      for actionIteration in range(self.boardSize ** 2):
        actions.append(0)

      self.table.append(actions)

  # takes a 1d array representing the board looking like this
  # 0 | 1 | 2
  # 3 | 4 | 5
  # 6 | 7 | 8
  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
  # the values should be
  #   0  = empty field
  #   1 = player move
  #   2  = bot move
  def getIndex(self, board):
    for iteration in range(len(self.states)):
      if self.states[iteration] == board:
        return iteration

    index = len(self.states)
    self.states.append(board)

    return index

  def getOutput(self, board, training):
    tableIndex = self.getIndex(board)

    outputs = self.table[tableIndex]

    exploring = False
    if training == True:
      prob = random.randrange(100)

      if prob >= 75:
        exploring = True

    highestIndex = random.randrange(len(outputs))
    if exploring == False:
      for iteration in range(len(outputs)):
        if outputs[iteration] > outputs[highestIndex] + 0.1:
          highestIndex = iteration

    return highestIndex

  def getReward(result, movesMade):
    # a Draw
    if result == 0:
      return -40

    # a Win
    if result == 1:
      reward = 100
      extraMoves = movesMade - 3

      if extraMoves < 0:
        return reward

      reward = reward - (extraMoves * 5)

      return reward

    # a Loss
    if result == -1:
      reward = -100
      extraMoves = movesMade - 3

      if extraMoves < 0:
        return reward

      reward = reward + (extraMoves * 3)

      return reward

  def save(self):
    with open(self.tableFile, "wb") as f:
      pickle.dump(self.table, f)

    with open(self.statusFile, "wb") as f:
      pickle.dump(self.states, f)

  def load(self):
    try:
      with open(self.tableFile, "rb") as f:
        self.table = pickle.load(f)
    except:
      return False

    try:
      with open(self.statusFile, "rb") as f:
        self.states = pickle.load(f)
    except:
      return False

    return True

  def evaluate(self, board, outputIndex, reward):
    state = self.getIndex(board)
    currentQValue = self.table[state][outputIndex]

    deltaValue = reward + (self.discountRate * self.maxReward) - currentQValue
    deltaQ = self.learningRate * deltaValue
    newQValue = currentQValue + deltaQ

    self.table[state][outputIndex] = newQValue