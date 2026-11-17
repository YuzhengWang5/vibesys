"""The multi search as a story: measure the input, play rounds one after another, name the winner.

This is the whole search, top to bottom. Each round is a task in a group of one: the root decides
only when to start the next and when the search is over. `SerialRun` holds what the search knows
and folds the finished rounds into it, and is shared with the single search. An operator stop
reaches the round (it settles what it started); the root sees it as `StopRequested`, begins no
more, and returns. The winner is named as the search goes (`ctx.choose`), so a stop or a
deadline adopts what led at that moment.
"""

from vibesys.candidates.api import measure_input
from vibesys.orchestration.multi.search._config import MultiConfig
from vibesys.orchestration.multi.search._operations import OPERATIONS
from vibesys.orchestration.multi.search._roles import ROLES
from vibesys.orchestration.multi.search._round import MultiRound, play_round
from vibesys.search_spans.api import SPANS, adoption_label
from vibesys.serial_search.api import ROADMAP, RoundDone, SerialRun
from vs_core.api.search import INPUT, Flow, NoWinner, StopRequested, TaskStopped, search


@search("multi", config=MultiConfig, operations=OPERATIONS, roles=ROLES, owned=ROADMAP, spans=SPANS)
async def multi(ctx: Flow, config: MultiConfig) -> NoWinner | None:
    """Measure the input, play the rounds in order within budget, name the best reviewed candidate."""
    ctx.checkpoint(config)
    ctx.fall_back_to_input(adoption_label(INPUT, None, None))
    run = SerialRun(config.hypotheses, ctx.facts.baseline, ctx.task.terms, await measure_input(ctx))
    ctx.fall_back_to_input(run.leading()[1])
    measured = unmeasured = 0
    async with ctx.group(limit=1) as rounds:
        while not run.over(stopping=ctx.stop_requested):
            if config.max_measured_rounds is not None and measured >= config.max_measured_rounds:
                break
            if (
                config.max_unmeasured_rounds is not None
                and unmeasured >= config.max_unmeasured_rounds
            ):
                return NoWinner(reason="Consecutive unmeasured round budget exhausted")
            begin = run.start()
            task = rounds.start(f"{begin.number:06d}", play_round, MultiRound.of(begin, config))
            match await ctx.first(rounds, ctx.stopped):
                case RoundDone() as done:
                    run.finished(done, task.span)
                    ctx.choose(*run.leading())
                    if done.closed is not None and done.closed.passed and done.closed.metrics:
                        measured += 1
                        unmeasured = 0
                    else:
                        unmeasured += int(done.counted)
                case TaskStopped():
                    run.stopped = True
                case StopRequested():
                    pass  # the group has stopped the round
            ctx.publish(run.snapshot())
    if config.max_unmeasured_rounds is not None and unmeasured >= config.max_unmeasured_rounds:
        return NoWinner(reason="Consecutive unmeasured round budget exhausted")
    return run.ending()
