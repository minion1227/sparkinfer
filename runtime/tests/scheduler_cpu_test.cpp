// CPU test for the continuous-batching scheduler — pure host policy, no GPU needed.
// Covers priority ordering, the token-budget stop, the forward-progress guarantee
// (an oversized group must still be scheduled, never starved), and preemption order.

#include "sparkinfer/scheduler.h"
#include <cstdio>

using namespace sparkinfer;
#define CHECK(x) do{ if(!(x)){ printf("FAIL: %s (line %d)\n", #x, __LINE__); return 1; } }while(0)

static SequenceGroup grp(uint64_t id, int num_seqs, int priority) {
    return SequenceGroup{ id, num_seqs, /*max_new_tokens=*/16, priority };
}

int main() {
    // 1. Priority packing: higher-priority group is scheduled first, within budget.
    {
        Scheduler s(SchedulePolicy::PRIORITY, /*max_tokens_per_batch=*/8);
        s.add_sequence_group(grp(1, 2, /*prio=*/5));
        s.add_sequence_group(grp(2, 3, /*prio=*/9));
        ScheduleBatch b = s.schedule();
        CHECK(b.total_tokens == 5);
        CHECK(b.decode_seq_ids.size() == 5);
        // group 2 (priority 9) comes before group 1 (priority 5)
        CHECK(b.decode_seq_ids[0] == 2 && b.decode_seq_ids[1] == 2 && b.decode_seq_ids[2] == 2);
        CHECK(b.decode_seq_ids[3] == 1 && b.decode_seq_ids[4] == 1);
    }

    // 2. Token-budget stop: strict priority halts once the next group would overflow.
    {
        Scheduler s(SchedulePolicy::PRIORITY, /*max_tokens_per_batch=*/4);
        s.add_sequence_group(grp(1, 2, /*prio=*/5));   // lower priority, would push total to 5
        s.add_sequence_group(grp(2, 3, /*prio=*/9));   // higher priority, fits (3 <= 4)
        ScheduleBatch b = s.schedule();
        CHECK(b.total_tokens == 3);
        CHECK(b.decode_seq_ids.size() == 3);
        for (auto id : b.decode_seq_ids) CHECK(id == 2);  // only the higher-priority group ran
    }

    // 3. Forward-progress guarantee (the fixed bug): a group larger than the whole
    //    batch budget must still be scheduled — never an empty batch while work exists.
    {
        Scheduler s(SchedulePolicy::PRIORITY, /*max_tokens_per_batch=*/4);
        s.add_sequence_group(grp(7, 10, /*prio=*/1));   // 10 > 4 budget
        ScheduleBatch b = s.schedule();
        CHECK(!b.decode_seq_ids.empty());               // pre-fix: this was empty (starvation)
        CHECK(b.total_tokens == 10);
        CHECK(b.decode_seq_ids.size() == 10);
        for (auto id : b.decode_seq_ids) CHECK(id == 7);
    }

    // 4. Empty scheduler yields an empty batch.
    {
        Scheduler s(SchedulePolicy::PRIORITY, 8);
        ScheduleBatch b = s.schedule();
        CHECK(b.decode_seq_ids.empty());
        CHECK(b.total_tokens == 0);
    }

    // 5. Preemption evicts lowest-priority groups first, until enough is freed.
    {
        Scheduler s(SchedulePolicy::PRIORITY, 8);
        s.add_sequence_group(grp(1, 2, /*prio=*/5));
        s.add_sequence_group(grp(2, 3, /*prio=*/1));   // lowest priority -> first victim
        s.add_sequence_group(grp(3, 1, /*prio=*/9));
        auto victims = s.preempt(/*tokens_needed=*/3);
        CHECK(victims.size() == 1);
        CHECK(victims[0] == 2);                          // freeing 3 tokens from group 2 suffices
    }

    // 6. Preemption accumulates across groups when one is not enough.
    {
        Scheduler s(SchedulePolicy::PRIORITY, 8);
        s.add_sequence_group(grp(1, 2, /*prio=*/5));
        s.add_sequence_group(grp(2, 3, /*prio=*/1));
        s.add_sequence_group(grp(3, 1, /*prio=*/9));
        auto victims = s.preempt(/*tokens_needed=*/4);
        CHECK(victims.size() == 2);                      // group 2 (3) + group 1 (2) = 5 >= 4
        CHECK(victims[0] == 2 && victims[1] == 1);       // ascending priority order
    }

    printf("scheduler_cpu_test: OK\n");
    return 0;
}
