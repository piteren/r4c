'<!--SKIP_FIX-->'
## r4c - base classes

Envy and Actor are base r4c classes.

- **Envy** - 
- **Actor** -


### Envy

##### reset
The Actor is responsible for resetting the Envy after it reaches a terminal state, e.g., before calling ```run()```.


### Actor

The ```Actor``` has its policy, which is used to make an action based on a given observation.

### TrainableActor(Actor)

The ```TrainableActor``` uses the experience gained while playing to train its policy.  
The ```TrainableActor``` uses ```ExperienceMemory``` to store the experience.
The ```TrainableActor``` needs to implement:
- ```get_action()```
- ```_build_training_data()``` and ```_update()```
- ```save()``` and ```load()```


##### Reward

The ```Actor``` trains itself (its policy) using ```Envy``` observation data.
The reward returned by the ```Envy``` should be used as part of the observation and probably not treated as a direct reward signal used to train the Actor's policy.
The ```Actor``` should use any additional information from the observation to adjust or set its own value of reward used during training.
It may apply a discount, factor, normalization, moving average, etc., to the reward returned by the ```Envy``` or even define it from scratch.