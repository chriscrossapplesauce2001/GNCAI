# TODO

## Next Up
- [ ] Add episode termination when swarm centroid reaches the target point (instead of always running 500 steps)
- [ ] Add completion bonus reward for reaching the target
- [ ] Retrain with the new termination condition

## Future Ideas
- [ ] Tune reward weights (formation error still ~12, could increase W_FORMATION)
- [ ] Curriculum training (5 agents, obstacles)
- [ ] Train for more episodes if convergence is insufficient
