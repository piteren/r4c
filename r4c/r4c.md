## r4c - base classes

Envy and Actor are base r4c classes.

- **Envy** - 
- **Actor** -


### Envy

##### reset
Actor is responsible for resetting the Envy after reaching terminal state, e.g. before calling ```run()```


### Actor

```Actor``` has its policy, which is used to make an action for given observation.

### TrainableActor(Actor)

```TrainableActor``` uses experience received while playing to train its policy.  
```TrainableActor``` uses ```ExperienceMemory``` to store given experience.
```TrainableActor``` leaves to implement:
- ```_get_action()```
- ```_build_training_data()``` and ```_update()```
- ```save()``` and ```load()```


##### Reward

```Actor``` trains itself (policy) using ```Envy``` observation data.
Reward returned by the ```Envy``` should be used as a part of observation and probably
not treated as a direct reward signal used to train Actor policy.
```Actor``` should use any additional information from observation to adjust or set
its own value of reward used while training.
He may apply discount, factor, normalization, moving average etc.
to reward returned by ```Envy``` or even define it from scratch.