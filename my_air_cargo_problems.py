from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        def load_actions():
            loads = []
            for c in self.cargos:
                for a in self.airports:
                    for p in self.planes:
                        # For this piece of cargo, assert that it is at this airport
                        # and assert this a plane is also at this same airport
                        precond_pos = [
                            expr("At({}, {})".format(p, a)),
                            expr("At({}, {})".format(c, a))
                        ]
                        precond_neg = []
                        # Once this action is executed, the cargo is now in
                        # the plane and it is no longer at the airport
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem]
                        )
                        loads.append(load)
            return loads

        def unload_actions():
            unloads = []
            for c in self.cargos:
                for a in self.airports:
                    for p in self.planes:
                        # For this piece of cargo, assert that it is in this plane
                        # which is at this airport
                        precond_pos = [
                            expr("At({}, {})".format(p, a)),
                            expr("In({}, {})".format(c, p))
                        ]
                        precond_neg = []
                        # Once this action is executed, the cargo is now at the
                        # airport and no longer in the plane
                        effect_add = [expr("At({}, {})".format(c, a))]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem]
                        )
                        unloads.append(unload)
            return unloads

        def fly_actions():
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys
        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str):
        possible_actions = []
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            # For every possible action, determine if it is possible by assessing
            # every clause for this action against the current state.
            # This is done by determining that all positive preconditions are
            # met in this state and all negative preconditions are not.
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
            if is_possible:
                possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action):
        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)
        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str):
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        count = 0
        kb = PropKB()
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())
        # This function simply counts the number of clauses in the goal state
        # that have not yet been met in the current state. By assuming that
        # preconditions don't matter this gives us the minimum number of actions
        # that must be carried out in order to satisfy the goal conditions
        for clause in self.goal:
            if clause not in kb.clauses:
                count += 1
        return count


def air_cargo_p1():
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [
        expr('At(C1, SFO)'),
        expr('At(C2, JFK)'),
        expr('At(P1, SFO)'),
        expr('At(P2, JFK)')
    ]
    neg = [
        expr('At(C2, SFO)'),
        expr('In(C2, P1)'),
        expr('In(C2, P2)'),
        expr('At(C1, JFK)'),
        expr('In(C1, P1)'),
        expr('In(C1, P2)'),
        expr('At(P1, JFK)'),
        expr('At(P2, SFO)')
    ]
    init = FluentState(pos, neg)
    goal = [
        expr('At(C1, JFK)'),
        expr('At(C2, SFO)')
    ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2():
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [
        expr('At(C1, SFO)'),
        expr('At(C2, JFK)'),
        expr('At(C3, ATL)'),
        expr('At(P1, SFO)'),
        expr('At(P2, JFK)'),
        expr('At(P3, ATL)')
    ]
    neg = [
        expr('At(C1, JFK)'),
        expr('At(C1, ATL)'),
        expr('In(C1, P1)'),
        expr('In(C1, P2)'),
        expr('In(C1, P3)'),
        expr('At(C2, SFO)'),
        expr('At(C2, ATL)'),
        expr('In(C2, P1)'),
        expr('In(C2, P2)'),
        expr('In(C2, P3)'),
        expr('At(C3, JFK)'),
        expr('At(C3, SFO)'),
        expr('In(C3, P1)'),
        expr('In(C3, P2)'),
        expr('In(C3, P3)'),
        expr('At(P1, JFK)'),
        expr('At(P1, ATL)'),
        expr('At(P2, SFO)'),
        expr('At(P2, ATL)'),
        expr('At(P3, JFK)'),
        expr('At(P3, SFO)')
    ]
    init = FluentState(pos, neg)
    goal = [
        expr('At(C1, JFK)'),
        expr('At(C2, SFO)'),
        expr('At(C3, SFO)')
    ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3():
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [
        expr('At(C1, SFO)'),
        expr('At(C2, JFK)'),
        expr('At(C3, ATL)'),
        expr('At(C4, ORD)'),
        expr('At(P1, SFO)'),
        expr('At(P2, JFK)')
    ]
    neg = [
        expr('At(C1, JFK)'),
        expr('At(C1, ATL)'),
        expr('At(C1, ORD)'),
        expr('In(C1, P1)'),
        expr('In(C1, P2)'),
        expr('At(C2, SFO)'),
        expr('At(C2, ATL)'),
        expr('At(C2, ORD)'),
        expr('In(C2, P1)'),
        expr('In(C2, P2)'),
        expr('At(C3, JFK)'),
        expr('At(C3, SFO)'),
        expr('At(C3, ORD)'),
        expr('In(C3, P1)'),
        expr('In(C3, P2)'),
        expr('At(C4, JFK)'),
        expr('At(C4, ATL)'),
        expr('At(C4, SFO)'),
        expr('In(C4, P1)'),
        expr('In(C4, P2)'),
        expr('At(P1, JFK)'),
        expr('At(P1, ATL)'),
        expr('At(P1, ORD)'),
        expr('At(P2, SFO)'),
        expr('At(P2, ATL)'),
        expr('At(P2, ORD)')
    ]
    init = FluentState(pos, neg)
    goal = [
        expr('At(C1, JFK)'),
        expr('At(C3, JFK)'),
        expr('At(C2, SFO)'),
        expr('At(C4, SFO)')
    ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
