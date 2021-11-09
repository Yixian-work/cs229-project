import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting, RingBuilding, CircleBuilding
from geometry import Point
from graphics import Text, Point as pnt # very unfortunate indeed

## This file is originally created as CS237B class material, modified by Yixian ##
## The scenario is the car learns to reach the goal position while also reaching several waypoint positions##

MAP_WIDTH = 80
MAP_HEIGHT = 80
LANE_WIDTH = 8.8 # Modified Lane Width (Twice as large)
SIDEWALK_WIDTH = 2.0
LANE_MARKER_HEIGHT = 3.8
LANE_MARKER_WIDTH = 0.5
BUILDING_WIDTH = (MAP_WIDTH - 2*SIDEWALK_WIDTH - 2*LANE_WIDTH - LANE_MARKER_WIDTH) / 2.
OBS_POS = [(MAP_WIDTH/2 - (1/2)*LANE_WIDTH, MAP_HEIGHT/4), (MAP_WIDTH/2 + (1/2)*LANE_WIDTH, MAP_HEIGHT/2), (MAP_WIDTH/2 - (1/2)*LANE_WIDTH, MAP_HEIGHT*(3/4))]
GOAL_POS = (MAP_WIDTH/2, MAP_HEIGHT)

PPM = 5 # pixels per meter

class ObstacleAvoidanceScenario(gym.Env):
    def __init__(self):
        self.seed(0) # just in case we forget seeding
        
        self.init_ego = Car(Point(MAP_WIDTH/2., 0), heading = np.pi/2)
        self.init_ego.velocity = Point(0., 5)
        self.init_ego.min_speed = -10.
        self.init_ego.max_speed = 30.
        
        self.dt = 0.1
        self.T = 20
        
        self.reset()
        
    def reset(self):
        self.world = World(self.dt, width = MAP_WIDTH, height = MAP_HEIGHT, ppm = PPM)
        
        self.ego = self.init_ego.copy()

        # Random initialization reset (Heading diff)
        # self.ego.center = Point(BUILDING_WIDTH + SIDEWALK_WIDTH + 2 + np.random.rand()*(2*LANE_WIDTH + LANE_MARKER_WIDTH - 4), self.np_random.rand()* MAP_HEIGHT/10.)
        self.ego.heading += np.random.randn()*0.1
        self.ego.velocity += Point(0, self.np_random.randn()*2)

        self.obs = []
        self.obs.append(Point(OBS_POS[0][0], OBS_POS[0][1]))
        self.obs.append(Point(OBS_POS[1][0], OBS_POS[1][1]))
        self.obs.append(Point(OBS_POS[2][0], OBS_POS[2][1]))
        self.goal = Point(GOAL_POS[0], GOAL_POS[1])
        
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, MAP_HEIGHT), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, MAP_HEIGHT), 'gray64'))

        # lane markers on the road (More real world scenario)
        for y in np.arange(LANE_MARKER_HEIGHT/2., MAP_HEIGHT - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))

        # Building (Collision on the side and obstacles)
        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH, MAP_HEIGHT)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH, MAP_HEIGHT)))
        self.world.add(CircleBuilding(Point(OBS_POS[0][0], OBS_POS[0][1]), LANE_WIDTH/2., 'black'))
        self.world.add(CircleBuilding(Point(OBS_POS[1][0], OBS_POS[1][1]), LANE_WIDTH/2., 'black'))
        self.world.add(CircleBuilding(Point(OBS_POS[2][0], OBS_POS[2][1]), LANE_WIDTH/2., 'black'))

        # Painting of Target/Goal/Starting Position (Used for visualizing the waypoint following)
        self.world.add(Painting(Point(GOAL_POS[0], GOAL_POS[1]), Point(LANE_WIDTH*2, SIDEWALK_WIDTH*2), 'red'))
        self.world.add(Painting(Point(GOAL_POS[0], 0.), Point(LANE_WIDTH*2, SIDEWALK_WIDTH*2),'blue'))
        # Respawn car itself in map
        self.world.add(self.ego)

        return self._get_obs()
        
    def close(self):
        self.world.close()
        
    @property 
    def observation_space(self): # 5-dim state space
        low = np.array([0, 0, self.ego.min_speed, -np.pi/2, 0])
        high= np.array([MAP_WIDTH, MAP_HEIGHT, self.ego.max_speed, np.pi/2, 2*np.pi])
        return Box(low=low, high=high)

    @property
    def action_space(self): # 5 actions to choose
        return [(0.4, 0), (-0.4, 0), (0, 1), (0, -1), (0, 0)]

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property 
    def goal_reached(self):
        return np.abs(self.goal.y - self.ego.y) < SIDEWALK_WIDTH

    @property
    def collision_exists(self):
        return self.world.collision_exists()
        
    def step(self, action):
        self.ego.set_control(action[0],action[1])
        self.world.tick()
        
        return self._get_obs(), self._get_reward(), self.collision_exists or self.goal_reached or self.world.t >= self.T, {}
        
    def _get_reward(self): # Define Reward Here (Need to specify)
        if self.collision_exists:
            return -200
        elif self.goal_reached:
            return 200
        else:
            return -0.01*(self.ego.velocity.y - 10)**2 - 0.05 * self.ego.acceleration**2
        
        # if self.active_goal < len(self.targets):
        #     return -0.01*self.targets[self.active_goal].distanceTo(self.ego)
        # return -0.01*np.min([self.targets[i].distanceTo(self.ego) for i in range(len(self.targets))])
        
    def _get_obs(self): # Return Current State (5 dimensional stuff)
        return np.array([self.ego.center.x, self.ego.center.y, self.ego.velocity.y, self.ego.velocity.x, self.ego.heading])
        
    def render(self, mode='rgb'):
        self.world.render()
        
    def write(self, text): # this is hacky, it would be good to have a write() function in world class
        if hasattr(self, 'txt'):
            self.txt.undraw()
        self.txt = Text(pnt(PPM*(MAP_WIDTH - BUILDING_WIDTH+2), self.world.visualizer.display_height - PPM*10), text)
        self.txt.draw(self.world.visualizer.win)
