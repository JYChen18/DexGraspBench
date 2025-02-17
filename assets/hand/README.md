# Modification Clarification

### Shadow 
1. Remove the forearm and wrist, because they are more like parts of the arm.
2. Exclude the contact between some neighboring links, such as `palm` and `rh_ffproximal`, which are not excluded by default because of `rh_ffknuckle`.
3. Unify the forcerange and kp of different joints. kp is set to 5 because the object is heavy (object mass=100g).

### Allegro
1. Use kp=5.

### Leap
1. Use simplified visual mesh to speedup loading.

