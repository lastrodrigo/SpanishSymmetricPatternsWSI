This is the scoring used for senseval 2, taken originally from Joseph
Rosenzweig's scoring for senseval.   There are 2 scorers due to limitations in
the string table for the original scorer.  The original is scorer.c, and the
other is "score" (written in python, requires 2.0 or better).

The scoring process uses a subsumption table.  In practice, this did not get
used much, and when it was used, it didn't reflect the tasks too well -- in
the english tasks, the senses are grouped into disjoint sets, so they don't
really fit into a subsumption table.  Consequently, the scoring for the
english tasks only constituted the fine and coarse grained scores, as the mid
grained score wouldn't make any sense.  The swedish lexical sample task had
the problem that it used too coarse of a subsumption map for the scoring: any
key, answer combination that shared the same lemma would be considered
correct. 

There is a bug in the scoring software whereby when a system supplies an
answer for which there is no key, it's total attempted score increases thus
reducing it's precision score.  This may not have been a bug in the past, but
as there are many tasks which edited there corpora and answer keys in the
midst of the contest, this became an issue.


