# Register Bank Conflict Notes

- Register file is partitioned into banks; conflict occurs when multiple operands hit same bank.
- Conflicts can increase instruction latency or reduce throughput.
- Use compiler reports and Nsight metrics to detect stalls due to register bank conflicts.
- Mitigate by reordering instructions or adjusting register usage patterns.
