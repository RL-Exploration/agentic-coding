# Concept / Micro-Skill Map for 50 Coding Puzzles

## Per-Puzzle Categorization

| # | Puzzle ID | Difficulty | Concept / Row (Micro-Skill) | Why It Fits |
|---|---|---|---|---|
| 001 | drone_relay_delivery | Medium | **Dijkstra with Resource-Constrained State** | Builds a relay graph between charging stations, then runs Dijkstra where state = (node, remaining range). Tests augmented shortest-path with intermediate resource resets. |
| 002 | venture_capital_allocation | Medium | **Grouped Knapsack DP** | Classic grouped knapsack: for each startup, pick at most one tier (mutually exclusive choices) to maximize return under a budget constraint. |
| 003 | dna_sequence_assembly | Hard | **Bitmask DP over Overlap Graph (TSP-Variant)** | Computes pairwise overlaps, then finds shortest superstring via bitmask DP over all subsets — structurally identical to Traveling Salesman. |
| 004 | file_system_disk_usage | Medium | **Tree Aggregation via Path Prefix Parsing** | Parses Unix paths into an implicit tree, then aggregates file sizes bottom-up. Tests string splitting + hierarchical accumulation. |
| 005 | shift_scheduler | Easy | **Greedy Interval Scheduling (Activity Selection)** | Textbook greedy: sort intervals by end time, greedily pick non-overlapping ones. Tests the canonical activity selection algorithm. |
| 006 | spn_cbc_cipher_decrypt | Hard | **Bitwise Cipher Implementation (SPN Block Cipher)** | Requires implementing inverse S-box, inverse bit-permutation, key schedule, and CBC chaining — all at the bit/nibble level. Tests precise bitwise manipulation. |
| 007 | network_packet_bandwidth_window | Medium | **Sliding Window Sum on Sorted Sequence** | Classic sliding window on timestamp-sorted packets to find the window of duration W with maximum total bytes. |
| 008 | social_influence_clusters | Medium | **Union-Find with Aggregate Metadata Tracking** | Union-Find where each component tracks a spokesperson (max influence, min ID tiebreak). Tests maintaining auxiliary data across union operations. |
| 009 | build_system_soft_hard_deps | Hard | **Topological Sort with Multi-Type Dependency Propagation** | Combines hard-dependency topological ordering for scheduling with soft-dependency BFS for invalidation propagation. Tests multi-edge-type DAG traversal + critical path. |
| 010 | pipeline_pressure_spread | Medium | **Segment Tree with Lazy Propagation (Range Add + Range Min/Max)** | Range-add updates and range-min/max queries require a segment tree with lazy propagation. Tests the full lazy-propagation implementation. |
| 011 | trie_autocomplete_engine | Medium | **Trie Construction with Prefix-Based Top-K Retrieval** | Build a trie from frequency-weighted words, then for each prefix query, DFS from the prefix node and return top-k by frequency. |
| 012 | er_triage_queue | Easy | **Multi-Key Priority Queue Ordering** | Heap with composite keys (-severity, arrival_time, name). Tests constructing a correct multi-criteria comparator for heap extraction. |
| 013 | crossword_grid_filler | Hard | **Backtracking Constraint Satisfaction (Slot-Based Grid Filling)** | Extract crossword slots, then use backtracking to place words with intersection constraints. Tests pruning + constraint propagation in CSP. |
| 014 | ring_rotation_image_transform | Medium | **Matrix Ring Extraction and Circular Array Rotation** | Extract concentric rings from a matrix, rotate each ring's elements by a given offset, write back. Tests index arithmetic on 2D matrix layers. |
| 015 | hash_collision_optimizer | Medium | **Modular Hash Collision Analysis (Brute-Force Optimization)** | Iterate over candidate table sizes, compute `x % m` distribution, score pairwise collisions. Tests modular arithmetic + optimization over a search range. |
| 016 | drone_shortest_path_no_fly_zones | Hard | **Computational Geometry Shortest Path (Tangent Visibility Graph)** | Build a visibility graph using tangent lines between circular obstacles + start/end, then run Dijkstra on it. Tests tangent-point computation, segment-circle intersection, arc-length calculation. |
| 017 | satellite_comm_windows | Medium | **Weighted Job Scheduling DP with Binary Search** | Sort jobs by end time, DP where each job's value adds to the best compatible previous job found via binary search. Classic weighted interval scheduling. |
| 018 | ministack_language_interpreter | Medium | **Stack-Based Language Interpreter with Control Flow Parsing** | Implement a token-by-token interpreter with arithmetic ops, DUP/SWAP, and nested IFPOS/ELSE/ENDIF control flow. Tests parsing + stack machine execution. |
| 019 | terraced_garden_path | Easy | **BFS Shortest Path on Constrained Grid** | Standard BFS on a 2D grid where edges exist only between cells with elevation difference <= 1. Tests basic BFS with neighbor validation. |
| 020 | supply_truck_capacity | Medium | **Binary Search on Answer with Greedy Feasibility Check** | Binary search on truck capacity C; for each candidate, greedily simulate deliveries counting refills. Classic "binary search the answer" pattern. |
| 021 | password_policy_enumeration | Hard | **Multi-Dimensional Constrained Counting DP (Memoization)** | DP state = (position, last_class, run_length, per-class-usage-tuple). Tests memoized recursion over a complex multi-constraint state space. |
| 022 | audio_waveform_merge | Medium | **Two-Pointer Sorted Merge with Tolerance Matching** | Two-pointer merge of two sorted lists; samples within epsilon are combined, others pass through. Tests the two-pointer technique with approximate matching. |
| 023 | coin_line_strategy | Medium | **Game Theory DP (Optimal Turn-Based Prefix-Sum Strategy)** | Minimax DP: each player takes 1-3 coins from the left end; DP on suffix index with prefix sums to compute the advantage. |
| 024 | radio_frequency_assignment | Hard | **Graph Coloring with Pre-Assigned Constraints (Backtracking CSP)** | Assign colors (frequencies) to graph nodes with adjacency constraints and fixed pre-assignments. Backtracking with pruning for lexicographically smallest solution. |
| 025 | flooded_mine_navigation | Medium | **Dijkstra with Resource-Constrained State** | State = (station, oxygen_remaining). Flooded tunnels consume oxygen; air stations refill it. Same augmented-Dijkstra pattern as P001. |
| 026 | water_pipe_distribution | Medium | **Max-Flow Network (Ford-Fulkerson / BFS Augmenting Paths)** | Build a flow network with a super-sink, run Edmonds-Karp (BFS-based Ford-Fulkerson) to compute maximum water delivery. |
| 027 | parallel_task_splitter | Hard | **Interval DP with Recursive Divide-and-Conquer Cost** | DP over contiguous subranges: either execute sequentially or split into two parallel halves (adding overhead c). Tests interval DP with min-of-max recurrence. |
| 028 | memory_pool_linked_list | Easy | **Linked List Manipulation with Free-Slot Pool Management** | Maintain a singly linked list within a fixed-size slot array, tracking free slots and supporting alloc/free/read operations. |
| 029 | skyline_visible_building_pairs | Medium | **Monotonic Stack for Visible Pair Counting** | Use a decreasing monotonic stack to count all pairs of mutually visible buildings. Tests the monotonic stack invariant for inter-element visibility. |
| 030 | genomic_longest_k_repeat | Hard | **Suffix Array + LCP Array with Sliding Window Minimum** | Build suffix/LCP arrays, then slide a window of size k-1 over LCP to find the longest substring occurring >= k times. |
| 031 | territory_claim_resolver | Medium | **Convex Hull Construction + Point-in-Convex-Polygon Test** | Compute convex hull per faction, then classify each landmark as inside/on/outside each hull. Tests Andrew's monotone chain + cross-product point containment. |
| 032 | realtime_leaderboard_fenwick | Medium | **Fenwick Tree (BIT) for Dynamic Rank Queries** | Maintain a BIT indexed by score to answer "how many players have a strictly higher score" in O(log n) per query. |
| 033 | consistent_hash_ring_router | Hard | **Consistent Hashing Ring with Virtual Node Management** | Place virtual nodes via MD5 on a ring, use bisect for clockwise lookup, handle add/remove node and key migration. Tests hashing + sorted-container management. |
| 034 | wolfram_cellular_automaton | Medium | **Rule-Based 1D Cellular Automaton Simulation** | Decode an 8-bit rule number into a lookup table, simulate wrap-around grid updates for each generation. Tests bit-decoding + synchronous state update. |
| 035 | card_draw_or_hold | Easy | **Expected Value Computation for Single-Step Decision** | Compute the expected value of drawing a card (sum of non-bust outcomes / deck size) and compare to current score. Tests basic probability / EV reasoning. |
| 036 | fractal_similarity_dimension | Hard | **Numerical Root-Finding on Matrix Spectral Radius (Moran's Equation)** | Construct a transfer matrix M(D) for mutually recursive fractal rules, then binary-search for D where the spectral radius = 1. |
| 037 | multi_criteria_tournament_ranking | Medium | **Multi-Key Custom Comparator Sorting** | Rank contestants by total score, then podium count, then best-round rank, then name — multi-level tiebreaking. Same sorting pattern as P012 but on computed statistics. |
| 038 | resource_deadlock_detector | Medium | **Cycle Detection in Functional Graph (Floyd / DFS)** | Build a wait-for graph (each process waits for at most one resource), detect all nodes participating in cycles. |
| 039 | feature_flag_rollout_optimizer | Hard | **Bitmask DP over Set Partition (Subset Enumeration)** | `dp[mask] = min over submasks s of mask: dp[mask^s] + cost[s]`. Same bitmask-subset-enumeration DP pattern as P003. |
| 040 | event_timeline_compression | Medium | **Coordinate Compression with Difference Array Sweep** | Map large coordinates to compressed indices, apply +1/-1 at interval boundaries, sweep to produce activity counts. |
| 041 | seismic_intensity_sparse_table | Medium | **Sparse Table for Static Range Min/Max Queries** | Precompute sparse tables for range-max and range-min, answer each (peak, trough, intensity) query in O(1). |
| 042 | calendar_free_slot_finder | Easy | **Interval Clipping, Merging, and Gap Identification** | Clip intervals to a work window, merge overlapping ones, return gaps. Tests the full interval-merge pipeline. |
| 043 | kmp_plagiarism_detector | Medium | **KMP String Matching Algorithm** | For each length-k substring of the suspect, run KMP search against the source. Tests correct failure-function construction + linear-time matching. |
| 044 | clock_sync_congruence_protocol | Hard | **Chinese Remainder Theorem with Extended GCD** | Reduce pairwise clock synchronization to modular congruences, solve iteratively using CRT + extended Euclidean. |
| 045 | protocol_state_machine_validator | Medium | **Deterministic Finite Automaton (DFA) Simulation** | Build a transition table, step through message sequence, report accept/reject/error. Direct DFA implementation. |
| 046 | dots_and_boxes_minimax | Hard | **Minimax Search with Memoization (Combinatorial Board Game)** | Full minimax with alpha-beta or memoization on bitmask game state. Tests game tree search with "extra turn on box completion" rule. |
| 047 | rectangle_union_area_sweep | Medium | **Sweep Line with Coordinate Compression for Area Computation** | Sweep vertical events left-to-right, maintain active y-intervals with compressed coordinates, accumulate area. Same sweep-line family as P040. |
| 048 | power_of_two_choices_load_balancer | Medium | **Event-Driven Simulation with Queue State Tracking** | Process a sequence of request/complete events, maintain per-server queues, track peak load + assignments. Tests simulation bookkeeping. |
| 049 | wildfire_spread_prediction | Medium | **Multi-Source Dijkstra on Weighted Grid** | Multiple fire sources at time 0; cells have different ignition costs. Multi-source Dijkstra / priority-queue BFS on 2D grid. |
| 050 | browser_cache_eviction | Hard | **Multi-Policy Cache System Design (LRU/LFU/FIFO)** | Implement a cache supporting three eviction policies per domain, with global capacity and domain-level LRU for eviction ordering. Tests data structure design + policy dispatch. |


## Grouped Concept Rows (Same Underlying Micro-Skill)

Puzzles that test the **exact same** underlying micro-skill are grouped into a single "row."
If the model fails both puzzles in a paired row, that is a strong signal of a specific skill gap.

| Row | Concept / Micro-Skill | Puzzles | Difficulty Spread |
|-----|---|---|---|
| A | Dijkstra with Resource-Constrained State | 001, 025 | Medium, Medium |
| B | Bitmask DP over Subset Enumeration | 003, 039 | Hard, Hard |
| C | Backtracking Constraint Satisfaction | 013, 024 | Hard, Hard |
| D | Game Theory DP / Minimax | 023, 046 | Medium, Hard |
| E | Multi-Key Custom Comparator Sorting | 012, 037 | Easy, Medium |
| F | Sweep Line with Coordinate Compression | 040, 047 | Medium, Medium |
| G | Grid-Based BFS / Dijkstra | 019, 049 | Easy, Medium |
| H | Grouped Knapsack DP | 002 | Medium |
| I | Tree Aggregation via Path Prefix Parsing | 004 | Medium |
| J | Greedy Interval Scheduling | 005 | Easy |
| K | Bitwise Cipher Implementation | 006 | Hard |
| L | Sliding Window Sum | 007 | Medium |
| M | Union-Find with Metadata Tracking | 008 | Medium |
| N | Topological Sort with Multi-Type Dependencies | 009 | Hard |
| O | Segment Tree with Lazy Propagation | 010 | Medium |
| P | Trie-Based Top-K Prefix Retrieval | 011 | Medium |
| Q | Computational Geometry Shortest Path | 016 | Hard |
| R | Weighted Job Scheduling DP | 017 | Medium |
| S | Stack-Based Interpreter with Control Flow | 018 | Medium |
| T | Binary Search on Answer | 020 | Medium |
| U | Multi-Dimensional Constrained Counting DP | 021 | Hard |
| V | Two-Pointer Merge with Tolerance | 022 | Medium |
| W | Max-Flow Network | 026 | Medium |
| X | Interval DP (Divide-and-Conquer Cost) | 027 | Hard |
| Y | Linked List with Pool Management | 028 | Easy |
| Z | Monotonic Stack for Pair Counting | 029 | Medium |
| AA | Suffix Array + LCP + Sliding Window | 030 | Hard |
| AB | Convex Hull + Point-in-Polygon | 031 | Medium |
| AC | Fenwick Tree for Dynamic Rank Queries | 032 | Medium |
| AD | Consistent Hashing with Virtual Nodes | 033 | Hard |
| AE | Rule-Based Cellular Automaton Simulation | 034 | Medium |
| AF | Expected Value Decision | 035 | Easy |
| AG | Numerical Root-Finding (Spectral Radius) | 036 | Hard |
| AH | Cycle Detection in Functional Graph | 038 | Medium |
| AI | Sparse Table for Range Queries | 041 | Medium |
| AJ | Interval Clipping + Merging + Gap-Finding | 042 | Easy |
| AK | KMP String Matching | 043 | Medium |
| AL | Chinese Remainder Theorem + Extended GCD | 044 | Hard |
| AM | DFA Simulation | 045 | Medium |
| AN | Event-Driven Simulation with State Tracking | 048 | Medium |
| AO | Multi-Policy Cache System Design | 050 | Hard |

**Total: 50 puzzles across 37 distinct concept rows (6 paired, 31 singleton).**


## Skill Category Taxonomy (Higher-Level Grouping)

For broader analysis, the 37 rows cluster into these skill families:

| Skill Family | Rows | Count |
|---|---|---|
| **Graph Algorithms** (shortest path, flow, coloring, topo sort, cycle detection) | A, G, N, Q, W, AH | 6 |
| **Dynamic Programming** (knapsack, bitmask, interval, counting, game theory, job scheduling) | B, D, H, R, U, X | 6 |
| **Data Structures** (segment tree, Fenwick, trie, sparse table, union-find, monotonic stack, linked list) | M, O, P, Z, AC, AI, Y | 7 |
| **String Algorithms** (suffix array, KMP, overlap) | AA, AK | 2 |
| **Computational Geometry** (convex hull, visibility graph) | AB, Q | 2 |
| **Greedy / Two-Pointer / Sliding Window** | J, L, T, V | 4 |
| **Sweep Line / Coordinate Compression / Intervals** | AJ, F | 2 |
| **Number Theory / Modular Arithmetic** | AL, AG, K | 3 |
| **Simulation / State Machines / Interpreters** | AE, AM, AN, S | 4 |
| **System Design / Hashing** | AD, AO | 2 |
| **Sorting / Comparators** | E | 1 |
| **Probability / Expected Value** | AF | 1 |
| **Tree/Path Aggregation** | I | 1 |
