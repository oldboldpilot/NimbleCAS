% Straight-line integer arithmetic: the shape the TRITON target accepts.
%
% A Triton kernel is a flat block of lanes with no call stack, so the compiler refuses
% recursion there. fib.pl and countdown.pl therefore reach C++ and CUDA but not Triton, and
% this file is the straight-line arithmetic that reaches all three. A CALL is not refused --
% a non-recursive one is inlined -- and fee.pl is what exercises that.
%
% Called (+In, -Out): eval_poly(X, Y) with Y = X*X + 3*X + 5.
eval_poly(X, Y) :-
    T1 is X * X,
    T2 is 3 * X,
    Y is T1 + T2 + 5.
