---
layout: default
title: Status
---

## Project Summary
The game will set up a respawnable ghast floating in the sky around 20 blocks away from the agent. The ghast will constantly shoot explosive fireball to the agent in every 3 seconds. The fireball will create a small but temperory igniting zone when it explodes. The fireball, however, can be bounced back if the agent hits it right in front of his face with a sword. The goal of the project is to let agent learn how to survive under the fireball blast and find the best timing the hit back the fireball to kill the ghast.  The agent will need to deceide an action among swinging the sword once, waiting the fireball to get closer, or dodging the fireball based on the information input at each frame.

## Approach
We're using the reinforcement learning with PPO algorithm from the Rllib. 

$$ 
\hat q(s, a) = \hat qq(s, a) + \alpha[r + \gamma\max_a\hat q(s', a')- \hat q(s, a)] 
$$

There are 100 steps at each episode, and there are three avaiable actions for the agent at each step:
* Strafe right
* Strafe left
* Swing the sword once (attack)

The agent is rewarding mainly based on whether the agent successfully dodges a fireball (reward + 0.5), hits back a fireball (reward + 1), gets hit by a fireball (reward - 1), or kills the ghast (reward + 3). The reward is dettermined by the returned value of <ObservationFromNearbyEntities> tag: the agent can only see the coordinate of each fireball, while the rewarding system can see some extra information like the motion of both the living ghast and the fireballs. 

The rewarding system determins whether the agent has successfully dodged a fireball by comaring the z values of the fireball and the agent (i.e. check if the fireball flies pass the agent). Secondly, the rewarding system determins whether a fireball has been hit back by checking the z motion of the ball (because the fireballs are initially flying towards the agent, if there is a fireball that is flying in the opposite direction, it must be hit back by the agent). Lastly, the determining of whether the agent gets hit or kills a ghast is rather simple: just check the life they have after each round. To make things easier, we give agent the fire resistance at the begining of each game, so that the agent won't get extra punishments if it's ignited accidentally.
  
At each step, the agent gets a observation array indicating the postion of each fireball that is currenly flying towards it. So far, we simply construct the array by concatenating every fireball's x y z values, and we'll try to figure out a better way to turn the observation into an array.

Last but not least, the system will spawn a new ghast after any of the original ghasts has been killed, so that if the agent performs really well and kills all the ghasts quickly, instead of quitting and starting a new episode early, it can still recieve more rewards in the same episode. 

<img src="https://github.com/nuayoas/Ghast_Killer/blob/main/Capture.PNG"/>
