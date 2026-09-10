% An accumulator loop: the recursive call is the LAST goal and its result IS the head's
% result, so the compiler turns it into a loop rather than recursion.
count(0, A, A).
count(N, A, R) :-
    N > 0,
    N1 is N - 1,
    A1 is A + N,
    count(N1, A1, R).
