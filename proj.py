# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

import collections


class GhastKiller(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size_y = 25
        self.size_x = 10
        self.size_z = 20
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 5
        self.obs_size_x = 5
        self.obs_size_y = 5
        self.obs_size_z = 5
        self.max_episode_steps = 100 #100
        self.log_frequency = 10 #10

        self.action_dict = {
            0: 'movewest 1',  # Move one block forward
            1: 'moveeast 1',  # Turn 90 degrees to the right
            2: 'attack 1',  # Destroy block
        }

        # Rllib Parameters
        # continuous
        self.action_space = Box(low=np.array([-1, 0]), high=np.array([1, 1])) # [move attack]

        # discrete
        # self.action_space = Discrete(len(self.action_dict))

        self.observation_space = Box(0, 1, shape=(self.obs_size_x * self.obs_size_y * self.obs_size_z, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # GhastKiller Parameters
        self.obs = None
        self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.ghasts = collections.defaultdict(dict)
        self.fireballs = collections.defaultdict(dict)
        self.agentPos = (0.5, 21, -9.5)

    def reset(self):
        print(self.steps)
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()
        print("reset")

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action

        # Discrete
        # command = self.action_dict[action]
        # self.agent_host.sendCommand(command)
        # self.episode_step += 1
        # time.sleep(.2)

        # Continuous
        moveCommand = f"strafe {action[0]}"
        attackCommand = f"attack {round(action[1])}"
        # print(moveCommand, turnCommand, attackCommand, sep="\t")

        self.agent_host.sendCommand(moveCommand)
        # time.sleep(.2)
        
        

        self.agent_host.sendCommand(attackCommand)
        time.sleep(.2)
        self.episode_step += 1
        # print(self.episode_step)

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_break_action = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def get_mission_xml(self):
        entity = ''
        entity += '<DrawEntity x="0.5" y="21" z="15" type="Ghast" yaw="180"/>'
        # entity += '<DrawEntity x="4.5" y="21" z="15" type="Ghast" yaw="180"/>'
        # entity += '<DrawEntity x="-4.5" y="21" z="15" type="Ghast" yaw="180"/>'
        
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Diamond Collector</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>

                                ''' + \
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{2}' y2='{2}' z1='{-self.size_z}' z2='{self.size_z}' type='air'/>"+ \
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{1}' y2='{1}' z1='{-self.size_z}' z2='{self.size_z}' type='glowstone'/>"+ \
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{self.size_y}' y2='{self.size_y}' z1='{-self.size_z}' z2='{self.size_z}' type='sea_lantern'/>" + \
                                f"<DrawCuboid x1='{-self.size_x}' x2='{-self.size_x}' y1='{2}' y2='{self.size_y}' z1='{-self.size_z}' z2='{self.size_z}' type='sea_lantern'/>"+ \
                                f"<DrawCuboid x1='{self.size_x}' x2='{self.size_x}' y1='{2}' y2='{self.size_y}' z1='{-self.size_z}' z2='{self.size_z}' type='sea_lantern'/>" + \
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{2}' y2='{self.size_y}' z1='{self.size_z}' z2='{self.size_z}' type='bedrock'/>"+ \
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{2}' y2='{self.size_y}' z1='{-self.size_z}' z2='{-self.size_z}' type='bedrock'/>" +\
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{20}' y2='{20}' z1='{-11}' z2='{-7}' type='bedrock'/>" +\
                                f"<DrawCuboid x1='{-self.size_x}' x2='{-self.size_x}' y1='{21}' y2='{21}' z1='{-11}' z2='{-7}' type='bedrock'/>" +\
                                f"<DrawCuboid x1='{self.size_x}' x2='{self.size_x}' y1='{21}' y2='{21}' z1='{-11}' z2='{-7}' type='bedrock'/>" +\
                                f"<DrawCuboid x1='{-self.size_x+1}' x2='{self.size_x-1}' y1='{21}' y2='{21}' z1='{-11}' z2='{-7}' type='air'/>" +\
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{20}' y2='{20}' z1='{self.size_z-7}' z2='{self.size_z}' type='bedrock'/>" +\
                                f"<DrawCuboid x1='{-self.size_x}' x2='{self.size_x}' y1='{21}' y2='{21}' z1='{self.size_z-7}' z2='{self.size_z-7}' type='bedrock'/>" +\
                                entity +\
                                '''

                                

                               
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>CS175GhastKiller</Name>
                        <AgentStart>
                            <Placement x="0.5" y="22" z="-9.5" pitch="0" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_sword"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <ContinuousMovementCommands/>  
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>

                            <ObservationFromGrid>
                                <Grid name="blocks">
                                    <min x="-'''+str(int(self.obs_size_x/2))+'''" y="-1" z="0"/>
                                    <max x="'''+str(int(self.obs_size_x/2))+'''" y="0" z="'''+str(int(self.obs_size_z))+'''"/>
                                </Grid>
                            </ObservationFromGrid>


                            <ObservationFromNearbyEntities>
                                <Range name="eyesights" xrange="10" yrange="5" zrange="50" />
                                
                            </ObservationFromNearbyEntities>
                            

                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps*2)+'''" />
                            <AgentQuitFromTouchingBlockType>
                                <Block type="glowstone" />
                            </AgentQuitFromTouchingBlockType>

                            
							

                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'GhastKiller' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
            allow_break_action: <bool> whether the agent is facing a diamond
        """
            
        obs = np.zeros((self.obs_size_z*self.obs_size_y*self.obs_size_x, ))
        allow_break_action = False

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                eyesights = observations["eyesights"]
                
                # Update entities
                self.updateEntities(eyesights)
                
                break
        return obs, allow_break_action

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Diamond Collector')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value)) 


    def updateEntities(self, eyesights: dict) -> None:
        removedBalls = set(self.fireballs.keys())
        removedGhasts = set(self.ghasts.keys())
        for entity in eyesights:
            eid, name = entity['id'], entity['name']

            if name == "Ghast" and entity["life"] > 0:
                removedGhasts -= {eid}
                self.ghasts[eid] = { "life" : entity["life"] }

            elif name == "Fireball" and entity["z"] >= self.agentPos[2]:
                removedBalls -= {eid}
                self.fireballs[eid] = {
                                        "pos" : (entity["x"], entity["y"], entity["z"]),
                                        "motion" : (entity["motionX"], entity["motionY"], entity["motionZ"])
                                      }

        for b in removedBalls: del self.fireballs[b] 
        for g in removedGhasts: del self.ghasts[g]                          


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=GhastKiller, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
