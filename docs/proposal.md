---
layout: default
title: Proposal
---

## Summary of the Project

The game will set up a ghast (maybe more than one later?) that is floating in the sky, and Steve will be located in a ditch. The ghast will constantly shoot explosive fireball to Steve once the game started. The fireball will create a small but temperory igniting zone when it explodes. Steve can servive no more than five explosions. The fireball, however, can be bounced back if Steve hits it right in front of his face with a sword. The player will control Steve to survive under the blast and kill the ghast by hitting the fireball back. Thus, the AI agent will need to learn how to hit the fireball in a perfect angle and timing. Meanwhile, the agent will also need to deceide an action among swinging the sword once, waiting the fireball to get closer, or dodging the fireball based on the information input at each frame.

## AI/ML Algorithms

We're planning to use RL to train the agent. The agent will need to learn about the hitting angle, hitting timing, and player's movement. At the beignning, the agent has no idea about how to bounce back the fireball properly, and whether it should try to hit the fireball or simply run away from it under certain circumstances. Therefore we'll need to come up with a policy with different rewards and punishments to help the agent learn to survive & battle. For example, the agent will get one point for hitting back the fireball successfully, even more points for killing or almost hitting the ghast, and one point deducted for getting hit.

## Evaluation Plan

Our baseline for this project is to make the character stay alive and kill the ghast successfully. Once we achieved the baseline, we will consider to increase the number of grasps to increase the diffculty of this game and decrease the time of killing the ghast. To evaulate the success of the project, we will be looking at the success rate of bouncing back the fireballs, and the total time it takes to kill the ghast eventually. The agent is expected to make random choices at first, and as the learning continues, the success rate (i.e. how many fireballs are successfully bounced back) should be improved and thus the total time it takes to kill a ghast game should be shorter as well.

At the end, we expect the agent to be able to swing the sword and hit the fireball at most of the time (it will be really good if around 65-75% of the balls are bounced back), and the agent should survive at least 10-15 attacks before died. Like I said, we'll moniter the success rate and the surviving time contantly to verify if the algorithm is working.

## Scheduled Office Hour

Jan 27, 3:15pm

