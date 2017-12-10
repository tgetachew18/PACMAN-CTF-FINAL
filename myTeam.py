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
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveReflexAgent'):
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

class OffensiveAgent(CaptureAgent):
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
    CaptureAgent.registerInitialState(self, gameState) 
    '''
    Your initialization code goes here, if you need any.    '''
    self.beliefs = {} #distribution of each opponent
    self.opponents = self.getOpponents(gameState)
    self.NEGINF = float("-inf")
    self.INF = float("inf")
    self.available_food = [food for food in self.getFood(gameState).asList()]
    self.depth = 2 
    for opponent in self.opponents:
        self.beliefs[opponent] = util.Counter()
        opp_pos = gameState.getInitialAgentPosition(opponent)
        self.beliefs[opponent][opp_pos] = 1.0
        
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.myPoints = [p for p in self.legalPositions if gameState.isRed(p) == self.red]
    
    
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
      if opp_distribution.totalCount() == 0:
              self.initializeUniformly(opponent)
          else:
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
              
      allPossible.normalize()
      self.beliefs[opponent] = allPossible

  
  
  def isPacman(self, gameState, agent, agentpos):
      return gameState.isRed(agentpos) ^ gameState.isOnRedTeam(agent)
  
  def getProbableStates(self, gamestate, opponent):
      """Populates the 5 most probable states  for an opponent"""
      sortedKeys = self.beliefs[opponent].sortedKeys()
      noisy = gamestate.getAgentPosition(opponent)
      if noisy is not None:
          sortedKeys = noisy
      curr = 0
      while curr < len(sortedKeys) and curr <= 5:
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
          if action == 'Stop':continue #SDfasdfadfasdfasfdaf
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
    tried = []
    myPos = gameState.getAgentPosition(self.index)
    closestfood = None
    food = self.getFood(gameState).asList()
    if len(food):
        closestfood = min([self.distancer.getDistance(myPos,f) for f in food])
    maxes = []
    for move in gameState.getLegalActions(self.index):
        if move == 'Stop':continue
        state_prime = gameState.generateSuccessor(self.index, move)
        value = self.Min_Value(state_prime, self.opponents[0], self.depth)
        print 'move',move,value
        tried.append((move,value))
        if value > v:
            v = value
            best_move = move
            maxes = [best_move]
        elif value == v:
            maxes.append(move)

   
    print 'best move',best_move,v
    return random.choice(maxes)


  def distancetoOpp(self, gameState, pos, opponent):
      opp_pos = gameState.getAgentPosition(opponent)
      dist = None
      if opp_pos is None:
          opp_pos = self.beliefs[opponent].argMax()
      dist = self.distancer.getDistance(pos, opp_pos)
      return dist
 
      
  def evaluationFunction(self, gameState):
    myPos = gameState.getAgentPosition(self.index)
    mystate = self.getCurrentObservation()
    food = self.getFood(gameState).asList()
    carry =  gameState.getAgentState(self.index).numCarrying
    score = self.getScore(gameState)
    border = (gameState.data.layout.width/2, gameState.data.layout.height/2)
    safety = min(self.legalPositions, key = lambda x:self.getMazeDistance(x, border))
    safteydist = self.getMazeDistance(myPos, safety)
    'Features we considered important: capsules,closestghost,closestfood'
    
    #Closest Capsule
    capsules = self.getCapsules(gameState)
    capsule_dist = [self.distancer.getDistance(c, myPos) for c in capsules]
    caps_val = 0
    if len(capsule_dist) != 0:
        c = min(capsule_dist)
        caps_val = 40/(c+1.0)
    if len(self.getCapsules(mystate))- len(capsules) != 0:
        caps_val = 40
        
        
        
    #closest ghost
    ghostval = 0
    opp = min(self.opponents, key = lambda x:self.distancetoOpp(gameState, myPos, x)) #nearest opponent
    opp_pos = gameState.getAgentPosition(opp) #opponetn position
    dist = self.distancer.getDistance(myPos, opp_pos) * 1.0 #distance
    if self.isPacman(mystate, opp, opp_pos):
        #Chase unless we're scared
        scared = mystate.getAgentState(self.index).scaredTimer
        if scared == 0:
            #We're not scared/chase!
            ghostval = 30/(dist + 1)
        else:
            #We are scared Just run away for now.. (??????)
            ghostval = -35/(dist + 1)
    else: #The close opponent is a ghost
        scared = mystate.getAgentState(opp).scaredTimer
        if scared == 0:
            #They're not scared get away prop to how much food we're carrying           
            ghostval = -80*dist
            #if carry != mystate.getAgentState(self.index).numCarrying:
               # ghostval *= 100
            
        else:
            print 'SCared'
            #The opponent is scared
            ghostval = 35*dist
            previous , current = mystate.getAgentPosition(opp), gameState.getAgentPosition(opp)
            

    
    if dist > 4:
        ghostval = 0

    #Food stuff
    closestfood = None
    if len(food):
        closestfood = min([self.distancer.getDistance(myPos,f) for f in food])
    left = len(food)
    total = len(self.available_food) * 1.0
    foodval = (30.0/(closestfood + 1)) + 22.0*((total - left)/total)    
    if len(food) - len(self.getFood(mystate).asList()) != 0: #If I am able to eat food right away, I should do it
        foodval *= 100 #???????
    
    #Don't be too greedy
    #Go back when u have a decent amount 
    backval = 0 
    babies = 0
    if self.getScore(mystate) < 1:
        babies = 4
    else:
        babies = 2
        
    if carry > babies:
        return foodval + -50 * safteydist 
    
    
        
    
        

    return 2*score + foodval + ghostval + caps_val + backval

  





class DefensiveAgent(CaptureAgent):
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
    CaptureAgent.registerInitialState(self, gameState) 
    '''
    Your initialization code goes here, if you need any.    '''
    self.beliefs = {} #distribution of each opponent
    self.opponents = self.getOpponents(gameState)
    self.NEGINF = float("-inf")
    self.INF = float("inf")
    self.available_food = [food for food in self.getFood(gameState).asList()]
    self.depth = 2 
    for opponent in self.opponents:
        self.beliefs[opponent] = util.Counter()
        opp_pos = gameState.getInitialAgentPosition(opponent)
        self.beliefs[opponent][opp_pos] = 1.0
        
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.myPoints = [p for p in self.legalPositions if gameState.isRed(p) == self.red]
    
    
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
      if opp_distribution.totalCount() == 0:
          self.initializeUniformly(opponent)
      else:
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
              
      allPossible.normalize()
      self.beliefs[opponent] = allPossible

  
  
  def isPacman(self, gameState, agent, agentpos):
      return gameState.isRed(agentpos) ^ gameState.isOnRedTeam(agent)
  
  def getProbableStates(self, gamestate, opponent):
      """Populates the 5 most probable states  for an opponent"""
      sortedKeys = self.beliefs[opponent].sortedKeys()
      noisy = gamestate.getAgentPosition(opponent)
      if noisy is not None:
          sortedKeys = noisy
      curr = 0
      while curr < len(sortedKeys) and curr <= 5:
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
          if action == 'Stop':continue #SDfasdfadfasdfasfdaf
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
    tried = []
    myPos = gameState.getAgentPosition(self.index)
    closestfood = None
    food = self.getFood(gameState).asList()
    if len(food):
        closestfood = min([self.distancer.getDistance(myPos,f) for f in food])
    for move in gameState.getLegalActions(self.index):
        if move == 'Stop':continue
        state_prime = gameState.generateSuccessor(self.index, move)
        value = self.Min_Value(state_prime, self.opponents[0], self.depth)
        tried.append((move,value))
        if value > v:
            v = value
            best_move = move

   
    #print best_move
    return best_move
    return random.choice(actions)

  def evaluationFunction(self, gameState):#betterdefense
    myPos = gameState.getAgentPosition(self.index)
    food = self.getFoodYouAreDefending( gameState).asList()

    'Features we considered important:closestpacman,food defending length'
    #distances to closest ghost and pacman
    Secretdefense =[]#to get closest ghost
    opendefense =[]#to get closest pacman
    mystate = self.getCurrentObservation()
    pacmanval = 1
    for enemy in self.opponents:
      opp_pos = gameState.getAgentPosition(enemy)
      if not opp_pos is None:
        d = self.distancer.getDistance(myPos, opp_pos)
        if self.isPacman(gameState, enemy, opp_pos):
          opendefense.append(d)
        else:
          Secretdefense.append(d)
    #print"ghosts",len(Secretdefense),"Pacman",len(opendefense)
    closestpacman = 0 
    potentialpacman = 0 #closest ghost
    ScaredTimes=gameState.data.agentStates[self.index].scaredTimer
    if len(Secretdefense):
      potentialpacman = min(Secretdefense)
    if len(opendefense):
      closestpacman = min(opendefense)
      #am i scared
      if closestpacman<ScaredTimes:
        print "I am scared as fuck!!"
        pacmanval *= -100 #get the fuck out
      if potentialpacman<5: pacmanval += 50 #???

    #if no pacman then attack
    foodistance = []
    foodattack = self.getFood(gameState).asList()
    foodval=1
    if len(foodattack): 
      foodistance = []
      for dis in foodattack:
        y = self.distancer.getDistance(myPos,dis)
        foodistance.append(y)
      closestfood = min(foodistance)
      if len(foodattack) - len(self.getFood(self.getCurrentObservation()).asList()) != 0: #If I am able to eat food right away, I should do it
        foodval *= 200 #???????
    
    score = self.getScore(gameState)
    #trigger attack function
    if len(opendefense)<1:
      fake = 2*score - 5*closestfood  -10*len(foodattack) + 2*pacmanval - potentialpacman
      #print fake
    else:
      x=0
      otherteam = self.getOpponents(gameState)
      for enemy in otherteam:
        opp_pos = gameState.getAgentPosition(enemy)
        if self.isPacman(gameState, enemy, opp_pos):
          x+=1
      if len(opendefense) - x != 0: #If I am able to eat pacman right away, I should do it
        pacmanVal *= 200
      fake =10*pacmanval + 50*len(food) - 100*closestpacman -5*len(opendefense)

    #print fake
    return fake


class DefensiveReflexAgent(OffensiveAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = []
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    potential_invaders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
      
    if len(potential_invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in potential_invaders]
      features['potinvaderDistance'] = min(dists)
      
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'potinvaderDistance': -1}

