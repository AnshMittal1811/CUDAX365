# Hopper TMA Notes

- TMA provides bulk tensor memory copy operations from global to shared memory.
- Designed to reduce copy overhead and improve SM utilization.
- Supports multicast and layout transformations for tile transfers.
- Requires SM90 and compiler support for cp.async.bulk operations.
- Consider aligning tiles to cache lines and use barrier synchronization.
