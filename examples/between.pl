% A NONDETERMINISTIC predicate: between(+Low, +High, -Out) succeeds once for every integer from
% Low to High inclusive. It is what continuation-passing emission exists for.
%
% Compiled deterministically this predicate would be truncated to its first solution, which is a
% wrong answer dressed as a right one. Compiled with `cps` it takes a callback invoked once per
% solution, and returning false from that callback stops the enumeration.
%
%   prolog_compile examples/between.pl between iio cpp 64 cps
between(Low, High, Low) :- Low =< High.
between(Low, High, Out) :-
    Low < High,
    Next is Low + 1,
    between(Next, High, Out).
