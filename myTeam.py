# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from itertools import product
import distanceCalculator
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BayesianAgentone', second = 'BayesianAgenttwo'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BayesianAgentone(CaptureAgent):
  """
  A Bayesian belief distribution for where the opponents are 
  with majority functionality taken from previous project. 
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState) #?
    '''
    Your initialization code goes here, if you need any.    '''
    
    self.beliefs = {} #distribution of each opponent
    self.opponents = self.getOpponents(gameState)
    self.NEGINF = float("-inf")
    self.INF = float("inf")
    self.depth = 2 
    for opponent in self.opponents:
        self.beliefs[opponent] = util.Counter()
        opp_pos = gameState.getInitialAgentPosition(opponent)
        self.beliefs[opponent][opp_pos] = 1.0
        
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    
    
  def isTerminalState(self, gamestate, d):
      if d == 0 or  gamestate.isOver():
          return True
      return False


  def initializeUniformly(self, opponent):
      """Initialize opponent's corresponding prob distribution"""
      for pos in self.legalPositions:self.beliefs[opponent][pos] = 1.0
      self.beliefs[opponent].normalize()
           
  def observe(self, gameState, opponent, observation):
      noisyDistance = observation[opponent]
      opp_distribution = util.Counter()
      position = gameState.getAgentPosition(self.index)
      for p in self.legalPositions:
          trueDistance = util.manhattanDistance(position, p)
          if trueDistance <= 5: #We don't need it in this case
              opp_distribution[p] = 0.0
          else:
              opp_distribution[p] = self.beliefs[opponent][p] * gameState.getDistanceProb(trueDistance, noisyDistance)
      opp_distribution.normalize()
      self.beliefs[opponent] = opp_distribution
        
  
  def elapseTime(self, gameState, opponent):
      allPossible = util.Counter()      
      for oldPos in self.legalPositions:
          newPosDist = util.Counter()
          vecs = [(0, 0), (-1, 0), (0, -1), (0, 1), (1, 0)] #Vecs for possible moves
          for a, b in vecs:
              new_pos = oldPos[0] + a, oldPos[1] + b
              if new_pos in self.legalPositions:
                  newPosDist[new_pos] = 1.0
                  
          newPosDist.normalize()         
          for newPos, prob in newPosDist.items():
              allPossible[newPos] += self.beliefs[opponent][oldPos] * prob
              
      if allPossible.totalCount() == 0:
          self.initializeUniformly(opponent)
      else:
          allPossible.normalize()
          self.beliefs[opponent] = allPossible

  
  
  def isPacman(self, gameState, agent, agentpos):
      return gameState.isRed(agentpos) ^ gameState.isOnRedTeam(agent)
  
  def getProbableStates(self, gamestate, opponent):
      """Populates the 5 most probable states  for an opponent"""
      sortedKeys = self.beliefs[opponent].sortedKeys()
      curr = 0
      while curr <= 1:
          pos = sortedKeys[curr]
          if self.beliefs[opponent][pos] == 0:
              break
          # make new staste at this positon for the enemy 
          currState = gamestate.deepCopy()
          newoppstate = game.AgentState(game.Configuration(pos, 'Stop'), self.isPacman(currState, opponent, pos))
          currState.data.agentStates[opponent] = newoppstate
          yield currState
          curr += 1


  def Min_Value(self, gamestate, opponent, d):
    """Just model it as a bayes opponent"""
    if self.isTerminalState(gamestate, d):
      return self.evaluationFunction(gamestate)
    v = 0
    num_actions = 0.0
    for state in self.getProbableStates(gamestate, opponent):
        actions = []
        try:
            actions = state.getLegalActions(opponent)
        except:
            pass
        for action in actions:
            state_prime = state.generateSuccessor(opponent, action)
            if opponent == max(self.opponents): #I am the last opponent
                v += self.Max_Value(state_prime, d - 1)
            else:
                v += self.Min_Value(state_prime, opponent + 2, d - 1)
            num_actions += 1
    if num_actions == 0:
        return 0
    else:
        return v/num_actions

  def Max_Value(self, state, d):
      """This is this agent"""
      if self.isTerminalState(state, d):
          return self.evaluationFunction(state)
      v = self.NEGINF
      for action in state.getLegalActions(self.index):
          state_prime = state.generateSuccessor(self.index, action)
          v = max(v, self.Min_Value(state_prime, self.opponents[0], d))
      return v

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    #start = time.time()

    actions = gameState.getLegalActions(self.index)
    noisyDistances = gameState.getAgentDistances()
    for opponent in self.opponents:
            opp_pos = gameState.getAgentPosition(opponent)                
            if opp_pos is None:
                self.elapseTime(gameState, opponent)
                self.observe(gameState, opponent, noisyDistances )
            else:
                inf_belief = util.Counter()
                inf_belief[opp_pos] = 1.0
                self.beliefs[opponent] = inf_belief
    self.displayDistributionsOverPositions(self.beliefs.values())
    
    '''
    You should change this in your own agent.
    '''
    best_move = None
    v = self.NEGINF
    for move in gameState.getLegalActions(self.index):
        state_prime = gameState.generateSuccessor(self.index, move)
        value = self.Min_Value(state_prime, self.opponents[0], self.depth)
        if value > v:
            v = value
            best_move = move

    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    #print best_move
    return best_move
    return random.choice(actions)

  def evaluationFunction(self, gameState):
    myPos = gameState.getAgentPosition(self.index)
    food = self.getFood(gameState)

    'Features we considered important: capsules,closestghost,closestfood'
    #capsules
    caps = self.getCapsules(gameState)
    capsule = []
    c = 0
    if len(capsule):
      for c in caps:
        capsule.append(self.distancer.getDistance(c,myPos))
      c = min(capsule)

    #closest ghost
    places = []
    Scared = []
    Secretdefense =[]
    ghostval = 0
    for enemy in self.opponents:
      opp_pos = gameState.getAgentPosition(enemy)
      if not opp_pos is None:
        if not self.isPacman(gameState, enemy, opp_pos):
          d = self.distancer.getDistance(myPos, opp_pos)
          places.append(d)
          Scared.append(gameState.getAgentState(enemy).scaredTimer)#potentialbug if scaredmin is not closest ghost
        else:
          d = self.distancer.getDistance(myPos, opp_pos)
          Secretdefense.append(d)

    if len(places):
      closestghost = min(places)
      ScaredTimes = min(Scared)
      if closestghost <= 4 and ScaredTimes < 1:
        ghostval += -100 #get the fuck out
      elif closestghost <=6 and ScaredTimes < 1:
        ghostval += -9 #be cautious
      if Scared > 1:
        if closestghost<ScaredTimes: ghostval += 100 #chase after ghost
    #transport back
    'if we have higher score take back when we have 3?'
    'else only take back if we have like 5?'
    x = gameState.getAgentState(self.index).numCarrying
    '''
    initialopp = gameState.getInitialAgentPosition(self.getOpponents(gameState)[0])
    myinitial = gameState.getInitialAgentPosition(self.index)
    boardmiddle = (initialopp[0] + myinitial[0])/2,(initialopp[1] + myinitial[1])/2
    backs = self.distancer.getDistance(myPos,(boardmiddle[0],myPos[1]))'''
    myinitial = gameState.getInitialAgentPosition(self.index)
    back = self.distancer.getDistance(myPos,myinitial)
    if self.getScore(gameState)>3 and x>3:
      fake = 3*score  + 70*ghostval -  50*back
    if x>6:
      fake = 3*score - closestfood + 70*ghostval - 50*back
    else:    
      #eat food
      #if i am offense and Pacman is near me eat only if the other defense agent is a little far

      foodlist = food.asList()
      if len(foodlist): 
        foodistance = []
        for dis in foodlist:
          y = self.distancer.getDistance(myPos,dis)
          foodistance.append(y)
        closestfood = min(foodistance)
      closestmember = float("inf")
      #low key defend
      zz=0;
      if len(Secretdefense): 
        nearpac = min(Secretdefense)
        for member in self.getTeam(gameState):
          if member!=self.index:
            closestmember = min(closestmember,self.distancer.getDistance(myPos, gameState.getAgentPosition(member)))
            #print closestmember
            #print "pal"
        if nearpac <5 ^ closestmember > 10:
          zz=10000
      score = self.getScore(gameState)
      fake = 3*score - 10*closestfood + 70*ghostval - 1000*c -80*len(foodlist) + 50*zz
      #print fake
    return fake

class BayesianAgenttwo(CaptureAgent):
  """
  A Bayesian belief distribution for where the opponents are 
  with majority functionality taken from previous project. 
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState) #?
    '''
    Your initialization code goes here, if you need any.    '''
    
    self.beliefs = {} #distribution of each opponent
    self.opponents = self.getOpponents(gameState)
    self.NEGINF = float("-inf")
    self.INF = float("inf")
    self.depth = 2 
    for opponent in self.opponents:
        self.beliefs[opponent] = util.Counter()
        opp_pos = gameState.getInitialAgentPosition(opponent)
        self.beliefs[opponent][opp_pos] = 1.0
        
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    
    
  def isTerminalState(self, gamestate, d):
      if d == 0 or  gamestate.isOver():
          return True
      return False


  def initializeUniformly(self, opponent):
      """Initialize opponent's corresponding prob distribution"""
      for pos in self.legalPositions:self.beliefs[opponent][pos] = 1.0
      self.beliefs[opponent].normalize()
           
  def observe(self, gameState, opponent, observation):
      noisyDistance = observation[opponent]
      opp_distribution = util.Counter()
      position = gameState.getAgentPosition(self.index)
      for p in self.legalPositions:
          trueDistance = util.manhattanDistance(position, p)
          if trueDistance <= 5: #We don't need it in this case
              opp_distribution[p] = 0.0
          else:
              opp_distribution[p] = self.beliefs[opponent][p] * gameState.getDistanceProb(trueDistance, noisyDistance)
      opp_distribution.normalize()
      self.beliefs[opponent] = opp_distribution
        
  
  def elapseTime(self, gameState, opponent):
      allPossible = util.Counter()      
      for oldPos in self.legalPositions:
          newPosDist = util.Counter()
          vecs = [(0, 0), (-1, 0), (0, -1), (0, 1), (1, 0)] #Vecs for possible moves
          for a, b in vecs:
              new_pos = oldPos[0] + a, oldPos[1] + b
              if new_pos in self.legalPositions:
                  newPosDist[new_pos] = 1.0
                  
          newPosDist.normalize()         
          for newPos, prob in newPosDist.items():
              allPossible[newPos] += self.beliefs[opponent][oldPos] * prob
              
      if allPossible.totalCount() == 0:
          self.initializeUniformly(opponent)
      else:
          allPossible.normalize()
          self.beliefs[opponent] = allPossible

  
  
  def isPacman(self, gameState, agent, agentpos):
      return gameState.isRed(agentpos) ^ gameState.isOnRedTeam(agent)
  
  def getProbableStates(self, gamestate, opponent):
      """Populates the 5 most probable states  for an opponent"""
      sortedKeys = self.beliefs[opponent].sortedKeys()
      curr = 0
      while curr <= 1:
          pos = sortedKeys[curr]
          if self.beliefs[opponent][pos] == 0:
              break
          # make new staste at this positon for the enemy 
          currState = gamestate.deepCopy()
          newoppstate = game.AgentState(game.Configuration(pos, 'Stop'), self.isPacman(currState, opponent, pos))
          currState.data.agentStates[opponent] = newoppstate
          yield currState
          curr += 1


  def Min_Value(self, gamestate, opponent, d):
    """Just model it as a bayes opponent"""
    if self.isTerminalState(gamestate, d):
      return self.evaluationFunction(gamestate)
    v = 0
    num_actions = 0.0
    for state in self.getProbableStates(gamestate, opponent):
        actions = []
        try:
            actions = state.getLegalActions(opponent)
        except:
            pass
        for action in actions:
            state_prime = state.generateSuccessor(opponent, action)
            if opponent == max(self.opponents): #I am the last opponent
                v += self.Max_Value(state_prime, d - 1)
            else:
                v += self.Min_Value(state_prime, opponent + 2, d - 1)
            num_actions += 1
    if num_actions == 0:
        return 0
    else:
        return v/num_actions

  def Max_Value(self, state, d):
      """This is this agent"""
      if self.isTerminalState(state, d):
          return self.evaluationFunction(state)
      v = self.NEGINF
      for action in state.getLegalActions(self.index):
          state_prime = state.generateSuccessor(self.index, action)
          v = max(v, self.Min_Value(state_prime, self.opponents[0], d))
      return v

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    noisyDistances = gameState.getAgentDistances()
    for opponent in self.opponents:
            opp_pos = gameState.getAgentPosition(opponent)                
            if opp_pos is None:
                self.elapseTime(gameState, opponent)
                self.observe(gameState, opponent, noisyDistances )
            else:
                inf_belief = util.Counter()
                inf_belief[opp_pos] = 1.0
                self.beliefs[opponent] = inf_belief
    self.displayDistributionsOverPositions(self.beliefs.values())
    
    '''
    You should change this in your own agent.
    '''
    best_move = None
    v = self.NEGINF
    for move in gameState.getLegalActions(self.index):
        state_prime = gameState.generateSuccessor(self.index, move)
        value = self.Min_Value(state_prime, self.opponents[0], self.depth)
        if value > v:
            v = value
            best_move = move
    #print best_move
    return best_move
    #return random.choice(actions)

  def evaluationFunction(self, gameState):
    myPos = gameState.getAgentPosition(self.index)
    food = self.getFoodYouAreDefending( gameState).asList()

    'Features we considered important:closestpacman,food defending length'
    #closest ghost
    Secretdefense =[]
    opendefense =[]
    pacmanval = 0
    for enemy in self.opponents:
      opp_pos = gameState.getAgentPosition(enemy)
      if not opp_pos is None:
        d = self.distancer.getDistance(myPos, opp_pos)
        if self.isPacman(gameState, enemy, opp_pos):
          opendefense.append(d)
        else:
          Secretdefense.append(d)
    #am i scared
    closestpacman = 0
    potentialpacman = 0
    ScaredTimes=gameState.data.agentStates[self.index].scaredTimer
    if len(Secretdefense):
      potentialpacman = min(Secretdefense)
    if len(opendefense):
      closestpacman = min(opendefense)
      if ScaredTimes >0:
        pacmanval = -100 #get the fuck out
      if closestpacman>ScaredTimes: pacmanval = 1000 #chase after ghost
      if potentialpacman<10: pacmanval+= 200


    #if no pacman or ghosts near the center attack
    #write code for tracking items at the center
    foodistance = []
    foodattack = self.getFood(gameState).asList()
    x=0
    if len(foodattack): 
      foodistance = []
      for dis in foodattack:
        y = self.distancer.getDistance(myPos,dis)
        foodistance.append(y)
      closestfood = min(foodistance)
    score = self.getScore(gameState)
    #else trigger attack function?
    if closestpacman<1 ^ potentialpacman >8:
      fake = score - closestfood - potentialpacman  
    else:
      fake = 2*score + 100*pacmanval + 50*len(food) - 2*closestpacman

    #print fake
    return fake
