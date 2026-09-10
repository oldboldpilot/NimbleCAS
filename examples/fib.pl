% Fibonacci and greatest common divisor: the moded, deterministic, integer subset the
% compiler accepts. Both are called (+In, -Out).
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

gcd(A, 0, A) :- A >= 0.
gcd(A, B, G) :-
    B > 0,
    R is A mod B,
    gcd(B, R, G).
