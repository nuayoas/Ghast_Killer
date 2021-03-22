---
layout: default
title:  Home
---

[Our Final Model](https://github.com/nuayoas/Ghast_Killer/tree/main/RES/episdode_10s_norespwan)

## Project Summary

The ghast is a minecraft monster who usually spawns in the hell. When a player is inside its shooting range, the ghast will constantly shoot explosive fireballs to the player every 3 seconds. A fireball will create a small but temporary igniting zone when it explodes. The fireball, however, can be bounced back if the player hits it right in front of his face with a sword. If the player hits the fireball at the right angle, he can kill the ghast with the redirected fireball.

The game will set up a ghast floating in the sky around 20 blocks away from the agent. And the agent will be spawned on a platform with a diamond sword, with the ability to move left and right horizontally. The goal of the project is to make the agent learn how to survive under the fireball blast and find the best timing to hit back the fireball and eventually kill the ghast. The agent will need to decide an action among swinging the sword once, waiting for the fireball to get closer, or dodging the fireball based on the information input at each frame.

## Approaches

There are three available actions for the agent at each step:
* Strafe right
* Strafe left
* Swing the sword once (attack)

We get the position vectors of each entity from the <ObservationFromNearbyEntities> tag. The reward system determines whether the agent has successfully dodged a fireball by comparing the fireball's z values and the agent (i.e., check if the fireball flies past the agent). Secondly, the reward system determines whether a fireball has been hit back by checking the fireball's z motion (because the fireballs are initially flying towards the agent, if there is a fireball flying in the opposite direction, it must be redirected by the agent). Lastly, determining whether the agent gets damaged or kills a ghast is rather simple: check their existence after each round. To make things easier, we give the agent the fire resistance, so we know if the player has directly been hit by a fireball by monitoring his health.

Baseline: when the training just started, the agent is moving and attacking randomly, often gets hit by fireballs, and barely hits the target.


We're using the reinforcement learning with PPO (Proximal Policy Optimization) algorithm from the Rllib with the linear neural network as the underlying model. We also tried the customized simple rule-based model (i.e. creating some simple rules to guide agent’s decisions based on the position vector of the fireballs and the location of the player). However, even though the running time of the algorithm was improved under the rule-based model, it didn’t work as expected and the learning outcome was not as good as the neural network model. So we moved back to the neural network model and tried to optimize the agent’s learning from other aspects.

$$ 
\hat q(s, a) = \hat q(s, a) + \alpha[r + \gamma\max_a\hat q(s', a')- \hat q(s, a)] 
$$

As for the rewards, we initially set different scores for all of the following cases: dodge  a fireball (reward + 0.5), redirect a fireball (reward + 1), get hit by a fireball (reward - 1), kill the target (reward + 3). After testing a few different settings, we changed the rewards to be 0 for dodging, +5 for redirecting a fireball, +10 for killing the ghast, and -5 for getting hit. We removed the dodging reward because in the early stage, the agent moves left and right randomly, and sometimes even though the fireball is quite far away, the agent still chooses to move even further away. Remembering our ultimate goal is to kill the ghast, rewarding this kind of behavior isn’t going to help (so ideally, the type of dodge we want is to move a little bit when it’s too risky to try to hit back the fireball, but universally rewarding every successful dodge doesn’t make too much sense).

As for the observation space, initially we simply returned the captured position of the current fireball [x, y, z] (since the ghast is shooting fireballs with a constant rate of one per 3 seconds, the current fireball will arrive at the player before the next one comes out, meaning that the agent only need to consider the position of the current fireball). However, later we realized that also giving the agent the player’s coordinate should help it to learn the pattern faster, so we changed the observation space to be [x_fireball, y_fireball, z_fireball, x_player]. The reason we just include the x coordinate of the player is that the player can only move left and right, and the y and z values basically stay the same all the time.

As for the episode length, initially we set the length of each episode to be 100 steps and the ghast will be respawned as soon as the player killed it. We let it run for 10,000 steps and record the returns. 

![returns_status](https://user-images.githubusercontent.com/45502113/111943593-f2e6c380-8a92-11eb-8f32-7ec0239f6502.png)


The return is the score (rewards for dodging and killing, and the punishment for getting hit), so the higher the score is, the better the performance is. 

As you can see, after 10,000 steps, even though there is a visible upward trend, the improvement of the agent’s performance is still not obvious and the returns fluctuate quite a bit. Later we change each episode’s length to be 30 seconds, and again let it train for a while.

![returns](https://user-images.githubusercontent.com/45502113/111943617-fda15880-8a92-11eb-864b-c7a7b5081e5d.png)


We changed the rewards since this test, so the scale of scores varied a little from the previous graph. But you can still tell that the amount of improvement overtime is not ideal. 

Then we decided to stop respawning ghasts, and end the episode as soon as the player killed the target. This is because we realized the behavior of the ghast is not related to the previous one, meaning that the agent doesn’t need to plan beyond one ghast. We also keeped the maximum length of each episode to be 30 seconds, and let it run for a bit longer.


![returns](https://user-images.githubusercontent.com/45502113/111943637-098d1a80-8a93-11eb-8ab0-4ed57974a86c.png)


Since this time the length of each episode is not always the same, we update the returns list based on the number of steps: the graph above is the total score the agent got in each 300 steps. 

Now, you can see that the improvement overtime is a bit more obvious, but the curve still fluctuates quite a bit. And if you only pay attention to the first 10,000 - 20,000 steps, the pattern doesn’t change much compared to the previous graphs.

After discussing our concern with the professor, we realized that we can try to make the length of each episode even shorter, because the ghast always shoots fireballs with the constant rate and speed, meaning that the motion of each fireball is independent from others, so the agent shouldn’t need to plan too much in advance of each fireball. This time, we decreased the maximum duration of each episode to 10 seconds (which is about 2 or 3 fireballs in each episode), keeped other settings unchanged, and let it run for a while.

![b40265fe8c52b929f8f53f3e99bfc6c](https://user-images.githubusercontent.com/45502113/111942833-6c7db200-8a91-11eb-9cdc-c4fd96dfefc7.png)

This time, the learning outcome improved significantly. After around 80,000 steps, the agent is able to hit back most of the fireballs, and kill the target within the 10-second period. The agent was not only gaining more points (500 points vs. 150 points at step 60,000), but also picking up the pattern of the game much faster: the score reached the maximum after 60,000 steps.

We stopped the training after 100,000 steps, since there wasn’t too much room left for the further improvement (in terms of returns).

## Evaluation

We keep track of four values overtime as the evaluation: returns (rewards), number of ghasts killed (within every x steps), number of fireballs redirected (within every x steps), and number of times getting hit (within every x steps).

Initially, our episode has fixed length (100 steps), so we simply recorded the total number of killing/hitting back/getting hit and the reward of each episode, and then appended them to specific arrays (e.g. self.returns) at the end of each episode.

However, later we changed the length of the episode to be unequal (i.e. the episode ends as soon as the ghast has been killed), which means the total step of each episode varies as well. To solve the problem, we added a total step counter, and whenever self.total_steps % x == 0, we appended the evaluation values (current return, total number of killing / hitting back / getting hit) to the arrays and reset the evaluation values to be 0. 

After trying several different numbers, we set x to be 300 (i.e. the graphs below show the evaluation values of the first 300 steps, then the second 300 steps etc.) This is because if x is too small (e.g. 100) the range of the values is small: for example, an untrained agent kills 0 or 1 ghast within 100 steps, while a well-trained agent can at most kill 3-4 ghasts. But if x is too large (e.g. 800) there will not be enough available data points, especially in the early stages, which makes it difficult to evaluate the learning result of the agent.

Here are our best learning outcomes:


![f30e5a2a911af1ec041003e291676c7](https://user-images.githubusercontent.com/45502113/111942760-33ddd880-8a91-11eb-991c-edd1442438c4.png)


The number of ghasts killed within each 300 steps. At the beginning, since the agent was basically moving and attacking randomly, the number of ghats killed is quite low (2-3 in average, and there were a lot of times that the agent didn’t kill any of the ghasts). However, after 60,000 steps, the agent could always kill the ghast in each 10-sec episode (as you can see in our demonstration video).


![c942db50c3e11784a6cd3db09ad83aa](https://user-images.githubusercontent.com/45502113/111942771-40623100-8a91-11eb-9667-3c3b7abf7a8a.png)

The number of redirected fireballs within each 300 steps. Similarly, the agent did poorly at the beginning (can only hit back around 5 fireballs), while after 60,000 steps, the agent was able to redirect most of the fireballs (around 35 within 300 steps). The success rate increased significantly during the first 60,000 steps.

![966677da189a189b080966ca7a54d0d](https://user-images.githubusercontent.com/45502113/111942798-5243d400-8a91-11eb-9330-adbe28096977.png)

The number of getting hit within each 300 steps. The number of getting hit decreased from around 6 at the early stage, to around 1 at the end. The obvious declining trend shows the agent also learned how to avoid being hit from training.

![b40265fe8c52b929f8f53f3e99bfc6c](https://user-images.githubusercontent.com/45502113/111942833-6c7db200-8a91-11eb-9cdc-c4fd96dfefc7.png)

The return (reward) of each 300 steps. This one is like a summary of the above graphs, because the reward is calculated from the number of killing / redirecting balls / getting hit. It’s easy to tell the returns in between each 300 steps improved significantly during the first 60,000 steps. After that, since the agent was able to kill most of the ghasts, the return reached the maximum and therefore didn’t change much in the following 40,000 steps.


## Resources Used

* [Rllib doc](https://docs.ray.io/en/latest/rllib-algorithms.html)
* [Malmo doc](https://microsoft.github.io/malmo/0.17.0/Documentation/index.html)
* [XML schema doc](https://microsoft.github.io/malmo/0.17.0/Schemas/MissionHandlers.html)
* [CS175 assignment2.py](https://canvas.eee.uci.edu/courses/30925/files/11335632)
* https://openai.com/blog/openai-baselines-ppo/


