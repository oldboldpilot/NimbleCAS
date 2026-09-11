% Factorial with an accumulator, which is what a WIDTH test needs.
%
% fib/2 in fib.pl is naive and doubly recursive, so the inputs that would exercise 128-bit
% arithmetic -- fib(93) is the first Fibonacci number past 2^63 -- are unreachable in practice.
% The accumulator here is tail recursive, so the compiler turns it into a loop and 33 iterations
% reach 8.68e36, comfortably inside 128 bits and far outside 64.
%
% Called (+N, +Acc, -Result), so fact_acc(N, 1, F) is N!. The boundaries that matter:
%
%   20! =                     2432902008176640000  the largest that fits in 64 bits
%   21! =                    51090942171709440000  refused at 64, exact at 128
%   33! =  8683317618811886495518194401280000000  the largest that fits in 128 bits
%   34! = 295232799039604140847618609643520000000  refused at 128, exact at arbitrary precision
%
% A refusal is the honest answer, not a failure of the compiler: an unrepresentable result is
% reported rather than wrapped.
fact_acc(0, A, A).
fact_acc(N, A, R) :-
    N > 0,
    A1 is A * N,
    N1 is N - 1,
    fact_acc(N1, A1, R).
